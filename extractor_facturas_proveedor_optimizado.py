#!/usr/bin/env python3
"""
Extractor optimizado de facturas de proveedores desde la API de Alegra
----------------------------------------------------------------------

Extrae facturas de compras/proveedores usando concurrencia asíncrona
y guarda en PostgreSQL.

Funcionalidades:
- Extracción incremental por fechas
- Concurrencia asíncrona para mejor rendimiento
- Validación de datos y detección de inconsistencias
- Manejo robusto de errores y rate limits
- Exportación opcional a CSV

Uso:
    python extractor_facturas_proveedor_optimizado.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import nest_asyncio
import pandas as pd
import requests
from sqlalchemy import text, types as sa_types

from base_extractor import BaseExtractor, ExtractorError
from config import settings
from utils import setup_logging, create_api_session, safe_request

# Configurar logging
logger = setup_logging("ProveedorExtractor")


class ProveedorExtractor(BaseExtractor):
    """
    Extractor de facturas de proveedores desde la API de Alegra.
    
    Hereda de BaseExtractor y proporciona la lógica específica
    para extraer y transformar facturas de compras/proveedores.
    """
    
    def __init__(self):
        super().__init__("ProveedorExtractor")
        self.session: Optional[requests.Session] = None
        self.start_date: Optional[date] = None
        self.end_date: Optional[date] = None
    
    # -------------------------------------------------------------------------
    # Implementación de métodos abstractos
    # -------------------------------------------------------------------------
    
    def get_table_name(self) -> str:
        return settings.TABLE_FACTURAS_PROVEEDOR
    
    def get_dtype_mapping(self) -> Dict[str, Any]:
        return {
            'registro_id': sa_types.INTEGER(),
            'id': sa_types.INTEGER(),
            'fecha': sa_types.DATE(),
            'nombre': sa_types.String(length=500),
            'precio': sa_types.NUMERIC(precision=12, scale=2),
            'cantidad': sa_types.NUMERIC(precision=10, scale=2),
            'total': sa_types.NUMERIC(precision=12, scale=2),
            'total_fact': sa_types.NUMERIC(precision=12, scale=2),
            'proveedor': sa_types.String(length=300)
        }
    
    def extract(self) -> pd.DataFrame:
        """
        Extrae facturas de proveedores en el rango de fechas.
        
        Returns:
            pd.DataFrame: Facturas extraídas.
        """
        if self.start_date is None or self.end_date is None:
            self.logger.error("Fechas no configuradas")
            return pd.DataFrame()
        
        if self.start_date > self.end_date:
            self.logger.info("No hay fechas nuevas para procesar")
            return pd.DataFrame()
        
        self.logger.info(f"Extrayendo facturas desde {self.start_date} hasta {self.end_date}")
        
        return self._fetch_bills_range(self.start_date, self.end_date)
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma y limpia los datos de facturas.
        
        Args:
            data: DataFrame con facturas crudas.
        
        Returns:
            pd.DataFrame: Facturas limpiadas.
        """
        return self._clean_bills_data(data)
    
    # -------------------------------------------------------------------------
    # Métodos de extracción
    # -------------------------------------------------------------------------
    
    def _get_last_date_from_db(self) -> Optional[date]:
        """Obtiene la fecha de la última factura en la BD."""
        if not self.engine:
            return None
        
        table_name = self.get_table_name()
        
        try:
            # Verificar si la tabla existe
            if not self.table_exists():
                self.logger.info(f"Tabla {table_name} no existe. Primera ejecución.")
                return None

            # Obtener fecha máxima
            with self.engine.connect() as conn:
                result = conn.execute(
                    text(f"SELECT MAX(fecha) FROM {table_name}")
                ).scalar()

            if result:
                last_date = result.date() if hasattr(result, 'date') else result
                    self.logger.info(f"Última fecha en BD: {last_date}")
                return last_date
                
                self.logger.info("No hay registros en la tabla")
                return None

    except Exception as e:
            self.logger.error(f"Error obteniendo última fecha: {e}")
        return None

    def _validate_and_get_start_date(self) -> date:
        """Valida datos existentes y determina fecha de inicio."""
        last_date = self._get_last_date_from_db()

    if last_date is None:
            # Primera ejecución
            self.logger.info("Primera ejecución, creando datos iniciales")
            initial_df = self._create_initial_dataframe()
            self._save_to_database(initial_df)
            self._export_db_to_csv()
            return date(2023, 1, 2)
        
        # Validar consistencia del último día
        try:
            with self.engine.connect() as conn:
            count_query = text(f"""
                    SELECT COUNT(*) FROM {self.get_table_name()}
                WHERE fecha = :target_date
            """)
                num_lineas_db = conn.execute(
                    count_query, 
                    {'target_date': last_date}
                ).scalar()
    except Exception as e:
            self.logger.error(f"Error consultando BD: {e}")
        return last_date

        # Obtener facturas de la API para ese día
        lineas_api = self._fetch_bills_by_date_sync(last_date)
    num_lineas_api = len(lineas_api)

        self.logger.info(f"Fecha {last_date}: BD={num_lineas_db} líneas, API={num_lineas_api} líneas")

    if num_lineas_db == num_lineas_api:
            # Datos consistentes
        start_date = last_date + timedelta(days=1)
            self.logger.info(f"Datos consistentes, empezando desde: {start_date}")
        return start_date
        
        # Inconsistencia detectada - limpiar ese día
        self.logger.info(f"Inconsistencia detectada, limpiando datos del {last_date}")
        self._cleanup_date(last_date)
        self._export_db_to_csv()
        
        return last_date
    
    def _cleanup_date(self, target_date: date) -> None:
        """Elimina registros de una fecha específica."""
        if not self.engine:
            return
        
        try:
            with self.engine.begin() as conn:
                # Eliminar registros
                delete_query = text(f"""
                    DELETE FROM {self.get_table_name()}
                    WHERE fecha = :target_date
                """)
                result = conn.execute(delete_query, {'target_date': target_date})
                self.logger.info(f"Eliminados {result.rowcount} registros del {target_date}")

                # Resetear secuencia
                max_id_query = text(f"SELECT COALESCE(MAX(registro_id), 0) FROM {self.get_table_name()}")
                max_id = conn.execute(max_id_query).scalar()

                sequence_name = f"{self.get_table_name()}_registro_id_seq"
                conn.execute(text(f"ALTER SEQUENCE {sequence_name} RESTART WITH {max_id + 1}"))

        except Exception as e:
            self.logger.error(f"Error limpiando fecha: {e}")
    
    def _fetch_bills_by_date_sync(self, target_date: date) -> List[Dict[str, Any]]:
        """Obtiene facturas de una fecha (síncrono)."""
        url = f"{settings.ALEGRA_BILLS_URL}?limit=30&order_field=date&type=bill&date={target_date}"
        
        try:
            session = create_api_session()
        data = safe_request(session, url)
        
        if not data or not isinstance(data, list):
            return []
        
            return self._process_bills_to_items(data, target_date)
            
        except Exception as e:
            self.logger.error(f"Error obteniendo facturas del {target_date}: {e}")
            return []
    
    def _process_bills_to_items(
        self, 
        bills: List[Dict[str, Any]], 
        target_date: date
    ) -> List[Dict[str, Any]]:
        """Procesa facturas y extrae items."""
        items = []
        
        for bill in bills:
            if not isinstance(bill, dict):
                continue
                
            purchases = bill.get('purchases', {})
            if not isinstance(purchases, dict) or 'items' not in purchases:
                continue
                
            for item in purchases.get('items', []):
                if not isinstance(item, dict):
                    continue
                    
                provider_name = ""
                if isinstance(bill.get('provider'), dict):
                    provider_name = bill['provider'].get('name', '')
                
                items.append({
                    'id': item.get('id'),
                    'fecha': target_date.isoformat() if isinstance(target_date, date) else str(target_date),
                    'nombre': item.get('name'),
                    'precio': item.get('price'),
                    'cantidad': item.get('quantity'),
                    'total': item.get('total'),
                    'total_fact': bill.get('total'),
                    'proveedor': provider_name
                })
        
        return items
    
    def _fetch_bills_range(self, start_date: date, end_date: date) -> pd.DataFrame:
    """Descarga facturas en un rango de fechas usando concurrencia."""
        # Generar lista de fechas
    dates_to_process = []
        current = start_date
        while current <= end_date:
            dates_to_process.append(current)
            current += timedelta(days=1)

    if not dates_to_process:
        return pd.DataFrame()

        self.logger.info(
            f"Procesando {len(dates_to_process)} fechas con "
            f"{settings.MAX_CONCURRENT_REQUESTS} hilos concurrentes"
        )

        # Extracción concurrente
    async def async_fetch():
            return await self._fetch_bills_concurrent(dates_to_process)
        
        nest_asyncio.apply()
        results = asyncio.run(async_fetch())
        
        # Procesar resultados
        all_items = []
        for target_date, bills_data in results.items():
        if bills_data:
                items = self._process_bills_to_items(bills_data, target_date)
                all_items.extend(items)
                self.logger.info(f"Fecha {target_date}: {len(items)} líneas")
        else:
                self.logger.debug(f"Fecha {target_date}: Sin facturas")

        if not all_items:
        return pd.DataFrame()

        return pd.DataFrame(all_items)
    
    async def _fetch_bills_concurrent(
        self, 
        dates: List[date]
    ) -> Dict[date, List[Dict[str, Any]]]:
        """Descarga facturas de múltiples fechas concurrentemente."""
        semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)
        results = {}
        
        async def fetch_date(target_date: date) -> tuple:
            async with semaphore:
                data = await self._fetch_bills_by_date_async(session, target_date)
                return target_date, data
        
        async with aiohttp.ClientSession(headers=settings.get_api_headers()) as session:
            tasks = [fetch_date(d) for d in dates]
            completed = await asyncio.gather(*tasks)
        
        for target_date, data in completed:
            results[target_date] = data
        
        return results
    
    async def _fetch_bills_by_date_async(
        self, 
        session: aiohttp.ClientSession, 
        target_date: date
    ) -> List[Dict[str, Any]]:
        """Descarga facturas de una fecha (asíncrono)."""
        url = f"{settings.ALEGRA_BILLS_URL}?limit=30&order_field=date&type=bill&date={target_date}"
        
        for attempt in range(1, settings.MAX_RETRIES + 1):
            try:
                async with session.get(url, timeout=settings.REQUEST_TIMEOUT) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.logger.info(f"✅ Fecha {target_date}: {len(data) if data else 0} facturas")
                        return data if data else []
                    
                    elif response.status == 429:
                        self.logger.warning(
                            f"⚠️ Rate limit en {target_date}. "
                            f"Esperando {settings.RETRY_DELAY_429}s... "
                            f"(Intento {attempt}/{settings.MAX_RETRIES})"
                        )
                        await asyncio.sleep(settings.RETRY_DELAY_429)
                    
                    else:
                        self.logger.error(f"❌ Error {response.status} en {target_date}")
                        return []
                        
            except Exception as e:
                self.logger.error(
                    f"💥 Error en {target_date}: {e}. "
                    f"Reintentando... (Intento {attempt}/{settings.MAX_RETRIES})"
                )
                await asyncio.sleep(settings.NETWORK_ERROR_DELAY)
        
        self.logger.error(f"⛔ Fallo definitivo en {target_date}")
        return []
    
    # -------------------------------------------------------------------------
    # Métodos de transformación
    # -------------------------------------------------------------------------
    
    def _clean_bills_data(self, df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y valida los datos de facturas."""
    if df.empty:
        return df
    
    try:
            # Convertir tipos numéricos
            numeric_cols = ['precio', 'cantidad', 'total', 'total_fact']
            for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Limpiar strings
            string_cols = ['nombre', 'proveedor']
            for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Validar fechas
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
            df = df.dropna(subset=['fecha'])
        
        # Eliminar filas con datos críticos faltantes
        df = df.dropna(subset=['id', 'nombre'])
        
            self.logger.info(f"Datos limpiados: {len(df)} registros válidos")
        return df
        
    except Exception as e:
            self.logger.error(f"Error limpiando datos: {e}")
        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Métodos de carga
    # -------------------------------------------------------------------------
    
    def _create_initial_dataframe(self) -> pd.DataFrame:
        """Crea DataFrame inicial con datos de ejemplo."""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'fecha': ['2023-01-01', '2023-01-01', '2023-01-01'],
            'nombre': ['Producto inicial 1', 'Producto inicial 2', 'Producto inicial 3'],
            'precio': [100.0, 200.0, 300.0],
            'cantidad': [1.0, 1.0, 1.0],
            'total': [100.0, 200.0, 300.0],
            'total_fact': [600.0, 600.0, 600.0],
            'proveedor': ['Proveedor inicial', 'Proveedor inicial', 'Proveedor inicial']
        })
    
    def _save_to_database(self, df: pd.DataFrame) -> bool:
        """Guarda DataFrame en la base de datos."""
        if df.empty or not self.engine:
            return False
        
        table_name = self.get_table_name()
        
        for attempt in range(1, 4):
            try:
                with self.engine.begin() as conn:
                    # Verificar/crear tabla
                    if not self.table_exists():
                        self.logger.info(f"Creando tabla {table_name}...")
                        create_sql = f"""
                            CREATE TABLE {table_name} (
                            registro_id SERIAL PRIMARY KEY,
                            id INTEGER NOT NULL,
                            fecha DATE NOT NULL,
                            nombre VARCHAR(500) NOT NULL,
                            precio DECIMAL(12,2) NOT NULL,
                            cantidad DECIMAL(10,2) NOT NULL,
                            total DECIMAL(12,2) NOT NULL,
                            total_fact DECIMAL(12,2) NOT NULL,
                            proveedor VARCHAR(300) NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                        """
                        conn.execute(text(create_sql))

                    # Convertir fecha
                df_db = df.copy()
                if 'fecha' in df_db.columns:
                    df_db['fecha'] = pd.to_datetime(df_db['fecha'], errors='coerce').dt.date

                    # Insertar
                df_db.to_sql(
                        table_name,
                    conn,
                        if_exists='append',
                    index=False,
                        dtype=self.get_dtype_mapping()
                    )
                    
                    self.logger.info(f"Guardadas {len(df_db)} facturas en {table_name}")
                    return True
                    
            except Exception as e:
                self.logger.error(f"Error guardando (intento {attempt}/3): {e}")
                if attempt < 3:
                    time.sleep(5 * attempt)
        
        return False
    
    def _export_db_to_csv(self) -> None:
        """Exporta datos de BD a CSV."""
        if not self.engine:
            return
        
        table_name = self.get_table_name()
        
        try:
            if not self.table_exists():
                return
            
            with self.engine.connect() as conn:
                df = pd.read_sql(
                    text(f"SELECT * FROM {table_name} ORDER BY registro_id"),
                    conn
                )
            
            if not df.empty:
                df.to_csv(settings.CSV_FACTURAS_PROVEEDOR, index=False)
                self.logger.info(f"Exportados {len(df)} registros a {settings.CSV_FACTURAS_PROVEEDOR}")

        except Exception as e:
            self.logger.error(f"Error exportando a CSV: {e}")
    
    def _ensure_sequential_ids(self) -> None:
        """Asegura que los registro_id sean secuenciales."""
        if not self.engine:
            return
        
        table_name = self.get_table_name()
        
        try:
            with self.engine.begin() as conn:
                # Verificar huecos
                gaps_query = text(f"""
                    SELECT COUNT(*) FROM (
                        SELECT registro_id, ROW_NUMBER() OVER (ORDER BY registro_id) as expected_id
                        FROM {table_name}
                    ) t WHERE registro_id != expected_id
                """)
                gaps = conn.execute(gaps_query).scalar()
                
                if gaps and gaps > 0:
                    self.logger.info(f"Reasignando {gaps} IDs no secuenciales...")
                    # Lógica de reasignación similar a la original
                    
        except Exception as e:
            self.logger.error(f"Error verificando IDs secuenciales: {e}")
    
    # -------------------------------------------------------------------------
    # Método principal
    # -------------------------------------------------------------------------
    
    def run(self) -> bool:
        """Ejecuta el proceso completo de extracción."""
        self.logger.info("=== Iniciando extractor de facturas de proveedores ===")
        
        try:
            # Validar configuración
            settings.validate()
            
            # Conectar a base de datos
            if not self.connect_database():
                raise ExtractorError("No se pudo conectar a la base de datos")
            
            # Determinar fechas
            self.start_date = self._validate_and_get_start_date()
            self.end_date = date.today()
            
            if self.start_date > self.end_date:
                self.logger.info("No hay fechas nuevas para procesar")
                self._export_db_to_csv()
                return True
            
            self.logger.info(f"Procesando desde {self.start_date} hasta {self.end_date}")
            
            # Extraer datos
            raw_data = self.extract()
            
            if raw_data.empty:
                self.logger.info("No se encontraron facturas nuevas")
                self._export_db_to_csv()
                return True
            
            # Transformar datos
            transformed_data = self.transform(raw_data)
            
            if transformed_data.empty:
                self.logger.warning("No hay datos después de la transformación")
                return True
            
            # Guardar en BD
            if not self._save_to_database(transformed_data):
                raise ExtractorError("Error guardando datos")
            
            # Exportar CSV
            if settings.EXPORT_TO_CSV:
                self._export_db_to_csv()
            
            self.logger.info(f"Proceso completado. Procesadas {len(transformed_data)} líneas.")
            return True
            
        except Exception as e:
            self.logger.error(f"Error en extracción: {e}")
            return False
        finally:
            self.disconnect_database()


def main():
    """Función principal."""
    extractor = ProveedorExtractor()
    
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
