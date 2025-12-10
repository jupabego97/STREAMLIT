#!/usr/bin/env python3
"""
Extractor concurrente de items desde la API de Alegra
------------------------------------------------------

Extrae el catálogo de productos/items desde la API de Alegra
usando concurrencia asíncrona y guarda en PostgreSQL.

Funcionalidades:
- Extracción completa del catálogo (reemplaza cada ejecución)
- Concurrencia asíncrona para mejor rendimiento
- Extracción de campos personalizados (código de barras, familia)
- Manejo robusto de errores y rate limits
- Items iniciales de servicio técnico

Uso:
    python items-extract.py
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import nest_asyncio
import pandas as pd
import requests
from sqlalchemy import text, types as sa_types

from base_extractor import BaseExtractor, ExtractorError
from config import settings
from utils import setup_logging

# Configurar logging
logger = setup_logging("ItemsExtractor")


class ItemsExtractor(BaseExtractor):
    """
    Extractor de items/productos desde la API de Alegra.
    
    Hereda de BaseExtractor y proporciona la lógica específica
    para extraer y transformar el catálogo de productos.
    """
    
    def __init__(self):
        super().__init__("ItemsExtractor")
        self.total_items: int = 0
    
    # -------------------------------------------------------------------------
    # Implementación de métodos abstractos
    # -------------------------------------------------------------------------
    
    def get_table_name(self) -> str:
        return settings.TABLE_ITEMS
    
    def get_dtype_mapping(self) -> Dict[str, Any]:
        return {
            'id': sa_types.INTEGER(),
            'nombre': sa_types.String(length=300),
            'codigo_barras': sa_types.String(length=50),
            'familia': sa_types.String(length=100),
            'precio': sa_types.NUMERIC(precision=12, scale=2),
            'fecha_inicial': sa_types.DATE(),
            'cantidad_disponible': sa_types.NUMERIC(precision=10, scale=2)
        }
    
    def extract(self) -> pd.DataFrame:
        """
        Extrae todos los items desde la API de Alegra.
        
        Returns:
            pd.DataFrame: Items extraídos.
        """
        # Obtener total de items
        self.total_items = self._get_total_items()
        
        if self.total_items == 0:
            self.logger.warning("No se encontraron items en la API")
            return pd.DataFrame()
        
        self.logger.info(f"Total de items en API: {self.total_items}")
        
        # Extraer items de manera concurrente
        return self._fetch_all_items_concurrent()
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma los datos crudos de items.
        
        Args:
            data: DataFrame con items crudos (lista de dicts).
        
        Returns:
            pd.DataFrame: Items transformados.
        """
        # Los datos ya vienen como lista de dicts de la extracción
        # Necesitamos procesarlos
        return data
    
    # -------------------------------------------------------------------------
    # Métodos de extracción
    # -------------------------------------------------------------------------
    
    def _get_total_items(self) -> int:
        """Obtiene el total de items desde la API."""
        try:
            url = f"{settings.ALEGRA_ITEMS_URL}?metadata=true"
            session = requests.Session()
            session.headers.update(settings.get_api_headers())
            
            response = session.get(url, timeout=settings.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            return int(data.get('metadata', {}).get('total', 0))
            
        except Exception as e:
            self.logger.error(f"Error obteniendo total de items: {e}")
            return 0
    
    def _fetch_all_items_concurrent(self) -> pd.DataFrame:
        """Descarga todos los items de manera concurrente."""
        async def async_fetch():
            return await self._fetch_items_async()
        
        try:
            nest_asyncio.apply()
            items = asyncio.run(async_fetch())
            
            if not items:
                return pd.DataFrame()
            
            # Procesar items
            return self._process_items(items)
            
        except Exception as e:
            self.logger.error(f"Error en extracción concurrente: {e}")
            return pd.DataFrame()
    
    async def _fetch_items_async(self) -> List[Dict[str, Any]]:
        """Descarga items de manera asíncrona."""
        semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)
        all_items = []
        
        async def fetch_page(offset: int) -> List[Dict[str, Any]]:
            async with semaphore:
                return await self._fetch_item_page(session, offset)
        
        async with aiohttp.ClientSession(headers=settings.get_api_headers()) as session:
            # Generar offsets
            offsets = list(range(0, self.total_items, settings.PAGE_SIZE))
            tasks = [fetch_page(offset) for offset in offsets]
            results = await asyncio.gather(*tasks)
        
        # Combinar resultados
        for page in results:
            if page:
                all_items.extend(page)
        
        return all_items
    
    async def _fetch_item_page(
        self, 
        session: aiohttp.ClientSession, 
        start: int
    ) -> List[Dict[str, Any]]:
        """Descarga una página de items."""
        url = f"{settings.ALEGRA_ITEMS_URL}?start={start}&limit={settings.PAGE_SIZE}&order_field=id"
        
        for attempt in range(1, settings.MAX_RETRIES + 1):
            try:
                async with session.get(url, timeout=settings.REQUEST_TIMEOUT) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.logger.info(f"✅ start={start} → {len(data)} items")
                        return data
                    
                    elif response.status == 429:
                        self.logger.warning(
                            f"⚠️ Rate limit en start={start}. "
                            f"Esperando {settings.RETRY_DELAY_429}s... "
                            f"(Intento {attempt}/{settings.MAX_RETRIES})"
                        )
                        await asyncio.sleep(settings.RETRY_DELAY_429)
                    
                    else:
                        self.logger.error(f"❌ Error {response.status} en start={start}")
                        return []
                        
            except Exception as e:
                self.logger.error(
                    f"💥 Error en start={start}: {e}. "
                    f"Reintentando... (Intento {attempt}/{settings.MAX_RETRIES})"
                )
                await asyncio.sleep(settings.NETWORK_ERROR_DELAY)
        
        self.logger.error(f"⛔ Fallo definitivo en start={start}")
        return []
    
    # -------------------------------------------------------------------------
    # Métodos de procesamiento
    # -------------------------------------------------------------------------
    
    def _process_items(self, items: List[Dict[str, Any]]) -> pd.DataFrame:
        """Procesa y limpia los datos de items."""
        rows = []
        
        for item in items:
            try:
                custom_fields = item.get('customFields', [])
                inventory = item.get('inventory', {})
                
                rows.append({
                    'id': int(item.get('id', 0)),
                    'nombre': item.get('name'),
                    'codigo_barras': self._extract_custom_field(custom_fields, 'Código de barras'),
                    'familia': self._extract_custom_field(custom_fields, 'FAMILIA'),
                    'precio': self._extract_price(item.get('price', [])),
                    'fecha_inicial': self._extract_inventory_date(inventory),
                    'cantidad_disponible': self._extract_inventory_quantity(inventory)
                })
                
            except Exception as e:
                self.logger.warning(f"Error procesando item {item.get('id')}: {e}")
        
        df = pd.DataFrame(rows)
        self.logger.info(f"Procesados {len(df)} items")
        return df
    
    @staticmethod
    def _extract_custom_field(
        custom_fields: List[Dict[str, Any]], 
        field_name: str
    ) -> Optional[str]:
        """Extrae un campo personalizado por nombre."""
        if not isinstance(custom_fields, list):
            return None
        
        for field in custom_fields:
            if field.get('name') == field_name:
                return field.get('value')
        return None
    
    @staticmethod
    def _extract_price(price_list: List[Dict[str, Any]]) -> Optional[float]:
        """Extrae el precio de la lista de precios."""
        if price_list and isinstance(price_list, list):
            try:
                return float(price_list[0].get('price', 0))
            except (ValueError, TypeError, IndexError):
                return None
        return None
    
    @staticmethod
    def _extract_inventory_date(inventory: Dict[str, Any]) -> Optional[str]:
        """Extrae la fecha inicial del inventario."""
        if not isinstance(inventory, dict):
            return None
        return inventory.get('initialQuantityDate')
    
    @staticmethod
    def _extract_inventory_quantity(inventory: Dict[str, Any]) -> Optional[float]:
        """Extrae la cantidad disponible del inventario."""
        if not isinstance(inventory, dict):
            return None
        
        qty = inventory.get('availableQuantity')
        try:
            return float(qty) if qty is not None else None
        except (ValueError, TypeError):
            return None
    
    # -------------------------------------------------------------------------
    # Métodos de carga
    # -------------------------------------------------------------------------
    
    def _create_initial_dataframe(self) -> pd.DataFrame:
        """Crea DataFrame con items iniciales de servicio técnico."""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'nombre': [
                'SERVICIO TECNICO',
                'SERVICIO TECNICO CONSOLA',
                'SERVICIO TECNICO IMPRESORA'
            ],
            'codigo_barras': ['0491', '0492', '0493'],
            'familia': ['SERVICIOS', 'SERVICIOS', 'SERVICIOS'],
            'precio': [0.0, 0.0, 0.0],
            'fecha_inicial': ['2023-01-02', '2023-01-02', '2023-01-02'],
            'cantidad_disponible': [0.0, 0.0, 0.0]
        })
    
    def _save_to_database(self, df: pd.DataFrame, mode: str = 'append') -> bool:
        """Guarda DataFrame en la base de datos."""
        if df.empty or not self.engine:
            return False
        
        try:
            df.to_sql(
                self.get_table_name(),
                self.engine,
                if_exists=mode,
                index=False,
                dtype=self.get_dtype_mapping(),
                method='multi'
            )
            self.logger.info(f"Guardados {len(df)} items ({mode})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error guardando items: {e}")
            return False
    
    def _export_to_csv(self) -> None:
        """Exporta items a CSV."""
        if not self.engine:
            return
        
        try:
            df = pd.read_sql_table(self.get_table_name(), self.engine)
            df.to_csv(settings.CSV_ITEMS, index=False)
            self.logger.info(f"Exportados {len(df)} items a {settings.CSV_ITEMS}")
        except Exception as e:
            self.logger.error(f"Error exportando a CSV: {e}")
    
    # -------------------------------------------------------------------------
    # Método principal
    # -------------------------------------------------------------------------
    
    def run(self) -> bool:
        """Ejecuta el proceso completo de extracción."""
        self.logger.info("=== Iniciando extractor de items (Alegra) ===")
        
        try:
            # Validar configuración
            settings.validate()
            
            # Conectar a base de datos
            if not self.connect_database():
                raise ExtractorError("No se pudo conectar a la base de datos")
            
            # Crear tabla con items iniciales (reemplaza)
            self.logger.info("Creando tabla con items iniciales...")
            initial_df = self._create_initial_dataframe()
            if not self._save_to_database(initial_df, mode='replace'):
                raise ExtractorError("Error creando tabla inicial")
            
            # Extraer items de la API
            self.logger.info("Descargando items de la API...")
            items_df = self.extract()
            
            if items_df.empty:
                self.logger.warning("No se obtuvieron items desde la API")
            else:
                # Agregar items extraídos
                if not self._save_to_database(items_df, mode='append'):
                    raise ExtractorError("Error guardando items")
            
            # Exportar a CSV si está configurado
            if settings.EXPORT_TO_CSV:
                self._export_to_csv()
            
            self.logger.info("=== Proceso finalizado exitosamente ===")
            return True
            
        except Exception as e:
            self.logger.error(f"Error en extracción: {e}")
            return False
        finally:
            self.disconnect_database()


def main():
    """Función principal."""
    extractor = ItemsExtractor()
    
    try:
        success = extractor.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Proceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
