#!/usr/bin/env python3
"""
Extractor de facturas de Alegra API a PostgreSQL
-------------------------------------------------

Extrae facturas de ventas desde la API de Alegra usando concurrencia
asíncrona y guarda en PostgreSQL.

Funcionalidades:
- Extracción incremental (solo facturas nuevas)
- Concurrencia asíncrona para mejor rendimiento
- Manejo robusto de errores y rate limits
- Exportación opcional a CSV

Uso:
    python extractor_facturas_alegra_sagrado.py
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import nest_asyncio
import pandas as pd
import requests
from sqlalchemy import text, types as sa_types

from base_extractor import BaseExtractor, ExtractorError
from config import settings
from utils import setup_logging, create_api_session

# Configurar logging
logger = setup_logging("FacturasExtractor")


class FacturasExtractor(BaseExtractor):
    """
    Extractor de facturas de ventas desde la API de Alegra.
    
    Hereda de BaseExtractor y proporciona la lógica específica
    para extraer y transformar facturas de ventas.
    """
    
    def __init__(self):
        super().__init__("FacturasExtractor")
        self.session: Optional[requests.Session] = None
    
    # -------------------------------------------------------------------------
    # Implementación de métodos abstractos
    # -------------------------------------------------------------------------
    
    def get_table_name(self) -> str:
        return settings.TABLE_FACTURAS
    
    def get_dtype_mapping(self) -> Dict[str, Any]:
        return {
            'id': sa_types.INTEGER(),
            'item_id': sa_types.INTEGER(),
            'fecha': sa_types.DATE(),
            'hora': sa_types.TIMESTAMP(),
            'nombre': sa_types.String(length=200),
            'precio': sa_types.FLOAT(),
            'cantidad': sa_types.INTEGER(),
            'total': sa_types.FLOAT(),
            'cliente': sa_types.String(length=200),
            'totalfact': sa_types.FLOAT(),
            'metodo': sa_types.String(length=50),
            'vendedor': sa_types.String(length=100)
        }
    
    def extract(self) -> pd.DataFrame:
        """
        Extrae facturas nuevas desde la API de Alegra.
        
        Returns:
            pd.DataFrame: Facturas extraídas (crudas).
        """
        # Determinar rango de IDs a extraer
        start_id = self._get_starting_invoice_id()
        end_id = self._get_latest_invoice_id()
        
        if start_id is None or end_id is None:
            self.logger.error("No se pudo determinar el rango de IDs")
            return pd.DataFrame()
        
        if start_id > end_id:
            self.logger.info("No hay nuevas facturas para procesar")
            return pd.DataFrame()
        
        self.logger.info(f"Extrayendo facturas del ID {start_id} al {end_id}")
        
        # Extraer facturas de manera concurrente
        return self._extract_invoices_concurrent(start_id, end_id)
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma los datos crudos de facturas.
        
        Args:
            data: DataFrame con facturas crudas.
        
        Returns:
            pd.DataFrame: Facturas transformadas a líneas de items.
        """
        if data.empty:
            return data
        
        # Limpiar datos
        cleaned_df = self._clean_invoice_data(data)
        
        if cleaned_df.empty:
            return cleaned_df
        
        # Transformar a líneas de items
        return self._transform_to_line_items(cleaned_df)
    
    # -------------------------------------------------------------------------
    # Métodos de extracción
    # -------------------------------------------------------------------------
    
    def _get_starting_invoice_id(self) -> Optional[int]:
        """Determina el ID desde donde iniciar la extracción."""
        last_id = self.get_max_id()
        if last_id:
            return last_id + 1
        
        self.logger.info("No hay facturas previas, iniciando desde ID 1")
        return 1
    
    def _get_latest_invoice_id(self) -> Optional[int]:
        """Obtiene el ID de la factura más reciente de la API."""
        try:
            # Determinar fecha de búsqueda
            if self.table_exists() and self.get_record_count() > 0:
                search_date = datetime.now() - timedelta(days=1)
            else:
                # Primera ejecución
                search_date = datetime(2022, 11, 1) + timedelta(days=30)
                self.logger.info("Primera ejecución, usando fecha inicial: 2022-11-01")
            
            url = (
                f"{settings.ALEGRA_INVOICES_URL}"
                f"?date_beforeOrNow={search_date.strftime('%Y-%m-%d')}"
                f"&order_direction=DESC&limit=1"
            )
            
            session = create_api_session()
            response = session.get(url, timeout=settings.REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            if data and len(data) > 0:
                latest_id = int(data[0]['id'])
                self.logger.info(f"Última factura en API: {latest_id}")
                return latest_id
            
            self.logger.warning("No se encontraron facturas recientes en la API")
            return None
            
        except Exception as e:
            self.logger.error(f"Error obteniendo última factura de API: {e}")
            return None
    
    def _extract_invoices_concurrent(
        self,
        start_id: int,
        end_id: int
    ) -> pd.DataFrame:
        """
        Extrae facturas de manera concurrente.
        
        Args:
            start_id: ID inicial.
            end_id: ID final.
        
        Returns:
            pd.DataFrame: Facturas extraídas.
        """
        async def async_extract():
            return await self._fetch_invoices_async(start_id, end_id)
        
        try:
            nest_asyncio.apply()
            df = asyncio.run(async_extract())
            self.logger.info(f"Total de facturas extraídas: {len(df)}")
            return df
        except Exception as e:
            self.logger.error(f"Error en extracción concurrente: {e}")
            return pd.DataFrame()
    
    async def _fetch_invoices_async(
        self,
        start_id: int,
        end_id: int
    ) -> pd.DataFrame:
        """
        Descarga facturas de manera asíncrona.
        
        Args:
            start_id: ID inicial.
            end_id: ID final.
        
        Returns:
            pd.DataFrame: Facturas combinadas.
        """
        semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)
        all_dfs = []
        
        async with aiohttp.ClientSession(headers=settings.get_api_headers()) as session:
            
            async def fetch_page(start: int) -> pd.DataFrame:
                async with semaphore:
                    return await self._fetch_invoice_batch(session, start)
            
            # Generar lista de offsets
            starts = list(range(start_id, end_id + 1, settings.PAGE_SIZE))
            tasks = [fetch_page(start) for start in starts]
            results = await asyncio.gather(*tasks)
        
        # Combinar resultados
        for df in results:
            if not df.empty:
                all_dfs.append(df)
        
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame()
    
    async def _fetch_invoice_batch(
        self,
        session: aiohttp.ClientSession,
        start: int
    ) -> pd.DataFrame:
        """
        Descarga una página de facturas.
        
        Args:
            session: Sesión aiohttp.
            start: Offset de inicio.
        
        Returns:
            pd.DataFrame: Facturas de la página.
        """
        url = (
            f"{settings.ALEGRA_INVOICES_URL}"
            f"?start={start}&order_direction=ASC&order_field=id&limit={settings.PAGE_SIZE}"
        )
        
        for attempt in range(1, settings.MAX_RETRIES + 1):
            try:
                async with session.get(url, timeout=settings.REQUEST_TIMEOUT) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.logger.info(f"✅ Página start={start} extraída con {len(data)} facturas")
                        return pd.DataFrame(data)
                    
                    elif response.status == 429:
                        self.logger.warning(
                            f"⚠️ Rate limit en start={start}. "
                            f"Esperando {settings.RETRY_DELAY_429}s... "
                            f"(Intento {attempt}/{settings.MAX_RETRIES})"
                        )
                        await asyncio.sleep(settings.RETRY_DELAY_429)
                    
                    else:
                        self.logger.error(f"❌ Error {response.status} en start={start}")
                        return pd.DataFrame()
                        
            except Exception as e:
                self.logger.error(
                    f"💥 Error en start={start}: {e}. "
                    f"Reintentando en {settings.NETWORK_ERROR_DELAY}s... "
                    f"(Intento {attempt}/{settings.MAX_RETRIES})"
                )
                await asyncio.sleep(settings.NETWORK_ERROR_DELAY)
        
        self.logger.error(f"⛔ Fallo definitivo en start={start}")
        return pd.DataFrame()
    
    # -------------------------------------------------------------------------
    # Métodos de transformación
    # -------------------------------------------------------------------------
    
    def _clean_invoice_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y procesa los datos de facturas.
        
        Args:
            df: DataFrame con facturas crudas.
        
        Returns:
            pd.DataFrame: Facturas limpiadas.
        """
        if df.empty:
            return df
        
        self.logger.info("Procesando datos de facturas...")
        
        # Columnas a eliminar
        columns_to_drop = [
            'observations', 'payments', 'subtotal', 'barCodeContent', 'total',
            'numberTemplate', 'dueDate', 'stamp', 'warehouse', 'term', 'anotation',
            'termsConditions', 'status', 'priceList', 'costCenter', 'paymentForm',
            'type', 'discount', 'tax', 'balance', 'decimalPrecision', 'operationType',
            'printingTemplate', 'station', 'retentions'
        ]
        existing_to_drop = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns=existing_to_drop, errors='ignore')
        
        # Extraer nombres de cliente y vendedor
        def extract_names(row):
            try:
                client_val = row.get('client')
                if isinstance(client_val, dict) and 'name' in client_val:
                    row['client'] = client_val['name']
                
                seller_val = row.get('seller')
                if isinstance(seller_val, dict) and 'name' in seller_val:
                    row['seller'] = seller_val['name']
                elif not seller_val:
                    row['seller'] = 'No se ha registrado un vendedor'
            except (KeyError, TypeError):
                pass
            return row
        
        df = df.apply(extract_names, axis=1)
        
        # Limpiar items
        def clean_items(row):
            try:
                items_val = row.get('items')
                if isinstance(items_val, list):
                    cleaned_items = []
                    for item in items_val:
                        if isinstance(item, dict):
                            cleaned_item = {
                                'id': item.get('id'),
                                'name': item.get('name', ''),
                                'price': item.get('price', 0),
                                'quantity': item.get('quantity', 0),
                                'total': item.get('total', 0)
                            }
                            cleaned_items.append(cleaned_item)
                    row['items'] = cleaned_items
            except (KeyError, TypeError):
                pass
            return row
        
        df = df.apply(clean_items, axis=1)
        return df
    
    def _transform_to_line_items(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma facturas a líneas de items individuales.
        
        Args:
            df: DataFrame con facturas limpiadas.
        
        Returns:
            pd.DataFrame: Líneas de items.
        """
        if df.empty:
            return pd.DataFrame()
        
        self.logger.info("Transformando a líneas de items individuales...")
        line_items = []
        
        for _, row in df.iterrows():
            try:
                items_val = row.get('items')
                if not isinstance(items_val, list):
                    continue
                
                for item in items_val:
                    # Valores con fallback
                    payment_method = row.get('paymentMethod')
                    metodo_val = payment_method if payment_method and pd.notna(payment_method) else 'Sin especificar'
                    cliente_val = row.get('client') or 'Sin especificar'
                    vendedor_val = row.get('seller') or 'No se ha registrado un vendedor'
                    nombre_val = item.get('name') or 'Sin nombre'
                    hora_val = row.get('datetime') or f"{row.get('date')} 00:00:00"
                    item_id_val = item.get('id') or 0
                    
                    line_item = {
                        'id': int(row['id']),
                        'item_id': int(item_id_val),
                        'fecha': row.get('date'),
                        'hora': hora_val,
                        'nombre': nombre_val,
                        'precio': float(item.get('price', 0)),
                        'cantidad': int(item.get('quantity', 0)),
                        'total': float(item.get('total', 0)),
                        'cliente': cliente_val,
                        'totalfact': float(row.get('totalPaid', 0) or 0),
                        'metodo': metodo_val,
                        'vendedor': vendedor_val
                    }
                    line_items.append(line_item)
                    
            except (KeyError, TypeError, ValueError) as e:
                self.logger.warning(f"Error procesando fila ID {row.get('id', 'desconocido')}: {e}")
                continue
        
        result_df = pd.DataFrame(line_items)
        self.logger.info(f"Generadas {len(result_df)} líneas de items")
        return result_df
    
    # -------------------------------------------------------------------------
    # Sobrescribir métodos de BaseExtractor
    # -------------------------------------------------------------------------
    
    def create_table_if_not_exists(self, create_sql: str = None) -> bool:
        """Crea la tabla facturas si no existe."""
        if not self.engine:
            self.logger.error("No hay conexión a la base de datos")
            return False
        
        try:
            with self.engine.connect() as conn:
                # Verificar si la tabla existe
                check_table_sql = """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = 'facturas'
                    );
                """
                table_exists = conn.execute(text(check_table_sql)).scalar()
                
                if not table_exists:
                    # Crear tabla nueva
                    create_table_sql = """
                        CREATE TABLE facturas (
                            indx SERIAL PRIMARY KEY,
                            id INTEGER NOT NULL,
                            item_id INTEGER NOT NULL,
                            fecha DATE NOT NULL,
                            hora TIMESTAMP NOT NULL,
                            nombre VARCHAR(200) NOT NULL,
                            precio FLOAT NOT NULL,
                            cantidad INTEGER NOT NULL,
                            total FLOAT NOT NULL,
                            cliente VARCHAR(200) NOT NULL,
                            totalfact FLOAT NOT NULL,
                            metodo VARCHAR(50) NOT NULL,
                            vendedor VARCHAR(100) NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                        
                        CREATE INDEX IF NOT EXISTS idx_facturas_id ON facturas(id);
                        CREATE INDEX IF NOT EXISTS idx_facturas_fecha ON facturas(fecha);
                        CREATE INDEX IF NOT EXISTS idx_facturas_item_id ON facturas(item_id);
                    """
                    conn.execute(text(create_table_sql))
                    conn.commit()
                    self.logger.info("Tabla facturas creada exitosamente")
                else:
                    # Verificar columna item_id
                    check_column_sql = """
                        SELECT EXISTS (
                            SELECT FROM information_schema.columns
                            WHERE table_schema = 'public'
                            AND table_name = 'facturas'
                            AND column_name = 'item_id'
                        );
                    """
                    column_exists = conn.execute(text(check_column_sql)).scalar()
                    
                    if not column_exists:
                        self.logger.info("Agregando columna item_id...")
                        conn.execute(text(
                            "ALTER TABLE facturas ADD COLUMN item_id INTEGER DEFAULT 0 NOT NULL;"
                        ))
                        conn.commit()
                    
                    # Crear índices si no existen
                    conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_facturas_id ON facturas(id);
                        CREATE INDEX IF NOT EXISTS idx_facturas_fecha ON facturas(fecha);
                        CREATE INDEX IF NOT EXISTS idx_facturas_item_id ON facturas(item_id);
                    """))
                    conn.commit()
                    self.logger.info("Tabla facturas verificada")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error creando/verificando tabla: {e}")
            return False
    
    def run(self) -> bool:
        """Ejecuta el proceso completo de extracción."""
        self.logger.info("=== Iniciando extracción de facturas Alegra ===")
        
        try:
            # Validar configuración
            settings.validate()
            
            # Conectar a base de datos
            if not self.connect_database():
                raise ExtractorError("No se pudo conectar a la base de datos")
            
            # Crear/verificar tabla
            if not self.create_table_if_not_exists():
                raise ExtractorError("No se pudo crear/verificar la tabla")
            
            # Extraer datos
            raw_data = self.extract()
            
            if raw_data.empty:
                self.logger.info("No hay nuevas facturas para procesar")
                if settings.EXPORT_TO_CSV:
                    self.export_to_csv(settings.CSV_FACTURAS)
                return True
            
            # Transformar datos
            transformed_data = self.transform(raw_data)
            
            if transformed_data.empty:
                self.logger.warning("No hay datos después de la transformación")
                return True
            
            # Cargar datos
            if not self.load(transformed_data):
                raise ExtractorError("Error cargando datos")
            
            # Exportar a CSV
            if settings.EXPORT_TO_CSV:
                self.export_to_csv(settings.CSV_FACTURAS)
            
            self.logger.info("=== Extracción completada exitosamente ===")
            return True
            
        except Exception as e:
            self.logger.error(f"Error en extracción: {e}")
            return False
        finally:
            self.disconnect_database()
    
    def export_to_csv(self, filename: str = None) -> bool:
        """Exporta facturas a CSV."""
        if not self.engine:
            return False
        
        csv_file = filename or settings.CSV_FACTURAS
        
        try:
            self.logger.info(f"Exportando datos a {csv_file}...")
            query = "SELECT * FROM facturas ORDER BY id, indx"
            df = pd.read_sql(query, self.engine)
            df_export = df.drop(columns=['indx', 'created_at'], errors='ignore')
            df_export.to_csv(csv_file, index=False)
            self.logger.info(f"Exportados {len(df_export)} registros a {csv_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error exportando a CSV: {e}")
            return False


def main():
    """Función principal."""
    extractor = FacturasExtractor()
    
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
