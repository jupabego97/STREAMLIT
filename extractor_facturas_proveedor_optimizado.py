#!/usr/bin/env python3
"""
Extractor optimizado de facturas de proveedores desde la API de Alegra
--------------------------------------------------------------------

▶ **Funcionalidades principales:**
   1. Extrae facturas de proveedores desde la API de Alegra usando concurrencia
   2. Procesa múltiples fechas simultáneamente con asyncio/aiohttp
   3. Manejo robusto de errores 429 con reintentos automáticos
   4. Procesa y limpia los datos de facturas con sus items
   5. Guarda en CSV y PostgreSQL con tipos de datos optimizados
   6. Manejo incremental: validación por fechas para evitar duplicados
   7. Iteración día por día para asegurar completitud

▶ **Mejoras respecto al notebook original:**
   - Manejo robusto de errores y reintentos
   - Configuración centralizada y flexible
   - Logging detallado para seguimiento
   - Validación de datos y tipos
   - Corrección de errores de comparación de fechas
   - Soporte para variables de entorno

Requisitos:
```bash
pip install pandas requests sqlalchemy psycopg2-binary python-dotenv aiohttp nest_asyncio
```
"""
from __future__ import annotations

import logging
import os
import sys
import time
import asyncio
import aiohttp
import nest_asyncio
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

# Importaciones opcionales para PostgreSQL
try:
    from sqlalchemy import create_engine, types as sa_types, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logging.warning("SQLAlchemy no está disponible. Solo se guardará en CSV.")


# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Config:
    """Configuración centralizada del extractor de facturas de proveedores."""

    # API Configuration
    base_url: str = "https://api.alegra.com/api/v1/bills"
    api_key_env: str = "ALEGRA_API_KEY"
    api_key_default: str = "bmFub3Ryb25pY3NhbHNvbmRlbGF0ZWNub2xvZ2lhQGdtYWlsLmNvbTphMmM4OTA3YjE1M2VmYTc0ODE5ZA=="

    # Request settings
    max_retries: int = 3
    backoff_factor: float = 1.5
    timeout_seconds: int = 30

    # Concurrency settings
    concurrent_requests: int = 4  # Número de peticiones simultáneas
    retry_delay_429: int = 60     # Segundos a esperar tras error 429
    network_error_delay: int = 5  # Segundos a esperar tras excepción de red
    max_retries_per_date: int = 5 # Máximo de reintentos por fecha

    # File settings
    csv_filename: str = "facturas_proveedor.csv"
    workspace_dir: str = "."
    require_csv: bool = True

    # Database settings
    db_url_env: str = "DATABASE_URL"
    db_table_name: str = "facturas_proveedor"

    # Processing settings
    log_level: int = logging.INFO


CFG = Config()

# ---------------------------------------------------------------------------
# Funciones asíncronas para extracción concurrente
# ---------------------------------------------------------------------------

async def fetch_bills_by_date_async(session, target_date: date) -> List[Dict[str, Any]]:
    """
    Extrae facturas de una fecha específica de manera asíncrona con reintentos.
    """
    url = f"{CFG.base_url}?limit=30&order_field=date&type=bill&date={target_date}"

    for attempt in range(1, CFG.max_retries_per_date + 1):
        try:
            async with session.get(url, headers={
                "accept": "application/json",
                "authorization": f"Basic {get_api_key()}"
            }, timeout=CFG.timeout_seconds) as response:

                status = response.status
                if status == 200:
                    data = await response.json()
                    logging.info(f"✅ Fecha {target_date} extraída con {len(data) if data else 0} facturas.")
                    return data if data else []
                elif status == 429:
                    logging.warning(
                        f"⚠️ Error 429 en fecha {target_date}. "
                        f"Esperando {CFG.retry_delay_429}s antes de reintentar... "
                        f"(Intento {attempt}/{CFG.max_retries_per_date})"
                    )
                    await asyncio.sleep(CFG.retry_delay_429)
                else:
                    logging.error(f"❌ Error {status} en fecha {target_date}. No se reintentará.")
                    return []

        except Exception as e:
            logging.error(
                f"💥 Excepción en fecha {target_date}: {e}. "
                f"Esperando {CFG.network_error_delay}s antes de reintentar... "
                f"(Intento {attempt}/{CFG.max_retries_per_date})"
            )
            await asyncio.sleep(CFG.network_error_delay)

    logging.error(f"⛔ Fallo definitivo en fecha {target_date} tras {CFG.max_retries_per_date} intentos.")
    return []


async def fetch_bills_concurrent(dates: List[date], concurrency: int = CFG.concurrent_requests) -> Dict[date, List[Dict[str, Any]]]:
    """
    Extrae facturas de múltiples fechas de manera concurrente usando asyncio.
    """
    nest_asyncio.apply()
    semaphore = asyncio.Semaphore(concurrency)
    results = {}

    async def bounded_fetch(target_date):
        async with semaphore:
            bills = await fetch_bills_by_date_async(session, target_date)
            return target_date, bills

    async with aiohttp.ClientSession() as session:
        tasks = [bounded_fetch(target_date) for target_date in dates]
        completed_tasks = await asyncio.gather(*tasks)

    # Convertir resultados a diccionario
    for target_date, bills in completed_tasks:
        results[target_date] = bills

    return results


def process_bills_data_async(bills_data: List[Dict[str, Any]], target_date: date) -> List[Dict[str, Any]]:
    """
    Procesa los datos de facturas para una fecha específica (versión síncrona para compatibilidad).
    """
    processed_items = []

    for bill in bills_data:
        if not isinstance(bill, dict):
            continue

        purchases = bill.get('purchases', {})
        if not isinstance(purchases, dict) or 'items' not in purchases:
            continue

        for item in purchases['items']:
            if not isinstance(item, dict):
                continue

            provider_name = ""
            if isinstance(bill.get('provider'), dict):
                provider_name = bill['provider'].get('name', '')

            processed_items.append({
                'id': item.get('id'),
                'fecha': target_date.isoformat(),  # Convertir date a string
                'nombre': item.get('name'),
                'precio': item.get('price'),
                'cantidad': item.get('quantity'),
                'total': item.get('total'),
                'total_fact': bill.get('total'),
                'proveedor': provider_name
            })

    logging.info(f"Procesados {len(processed_items)} items para fecha {target_date}")
    return processed_items

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO):
    """Configura el sistema de logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class ExtractorError(Exception):
    """Excepción personalizada para errores del extractor."""


def get_api_key() -> str:
    """Obtiene la API key desde variable de entorno o usa la por defecto."""
    api_key = os.getenv(CFG.api_key_env)
    if not api_key:
        logging.warning(f"Variable {CFG.api_key_env} no encontrada, usando clave por defecto")
        return CFG.api_key_default
    return api_key


def create_session(api_key: str) -> requests.Session:
    """Crea una sesión HTTP con autenticación configurada."""
    session = requests.Session()
    session.headers.update({
        "accept": "application/json",
        "authorization": f"Basic {api_key}"
    })
    return session


def safe_request(session: requests.Session, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Realiza una petición HTTP con reintentos y manejo de errores."""
    for attempt in range(1, CFG.max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=CFG.timeout_seconds)
            response.raise_for_status()
            
            if not response.text.strip():
                logging.warning("Respuesta vacía recibida")
                return []
                
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if attempt == CFG.max_retries:
                raise ExtractorError(f"Error en petición después de {CFG.max_retries} intentos: {e}")
            
            wait_time = CFG.backoff_factor ** attempt
            logging.warning(f"Intento {attempt} falló ({e}). Reintentando en {wait_time:.1f}s...")
            time.sleep(wait_time)


# ---------------------------------------------------------------------------
# Manejo de archivos CSV
# ---------------------------------------------------------------------------

def get_last_date_from_db(engine) -> Optional[date]:
    """Obtiene la fecha de la última factura desde la base de datos."""
    if engine is None:
        logging.info("No hay conexión a BD. Primera ejecución.")
        return None

    try:
        with engine.connect() as conn:
            # Verificar si la tabla existe
            table_exists_query = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = :table_name
                );
            """)

            table_exists = conn.execute(table_exists_query, {'table_name': CFG.db_table_name}).scalar()

            if not table_exists:
                logging.info(f"Tabla {CFG.db_table_name} no existe. Primera ejecución.")
                return None

            # Obtener la fecha máxima de la tabla
            max_date_query = text(f"SELECT MAX(fecha) as max_date FROM {CFG.db_table_name}")
            result = conn.execute(max_date_query).scalar()

            if result:
                last_date = result.date() if hasattr(result, 'date') else result
                logging.info(f"Última fecha en BD: {last_date}")
                return last_date
            else:
                logging.info("No hay registros en la tabla. Primera ejecución.")
                return None

    except Exception as e:
        logging.error(f"Error leyendo BD: {e}")
        return None


def ensure_sequential_ids(engine):
    """Asegura que los registro_id sean secuenciales sin huecos."""
    if engine is None:
        return

    try:
        with engine.begin() as conn:
            # Verificar si hay huecos en los IDs
            gaps_query = text(f"""
                SELECT COUNT(*) as gaps
                FROM (
                    SELECT registro_id, ROW_NUMBER() OVER (ORDER BY registro_id) as expected_id
                    FROM {CFG.db_table_name}
                ) t
                WHERE registro_id != expected_id
            """)

            gaps = conn.execute(gaps_query).scalar()

            if gaps > 0:
                logging.info(f"Detectados {gaps} huecos en registro_id. Reasignando IDs secuenciales...")

                # Crear tabla temporal con IDs secuenciales
                conn.execute(text(f"""
                    CREATE TEMP TABLE temp_facturas_proveedor AS
                    SELECT ROW_NUMBER() OVER (ORDER BY fecha, registro_id) as new_id,
                           id, fecha, nombre, precio, cantidad, total, total_fact, proveedor
                    FROM {CFG.db_table_name}
                    ORDER BY fecha, registro_id
                """))

                # Limpiar tabla original
                conn.execute(text(f"TRUNCATE TABLE {CFG.db_table_name}"))

                # Insertar con IDs secuenciales
                conn.execute(text(f"""
                    INSERT INTO {CFG.db_table_name} (registro_id, id, fecha, nombre, precio, cantidad, total, total_fact, proveedor)
                    SELECT new_id, id, fecha, nombre, precio, cantidad, total, total_fact, proveedor
                    FROM temp_facturas_proveedor
                """))

                # Resetear secuencia
                max_id_query = text(f"SELECT MAX(registro_id) FROM {CFG.db_table_name}")
                max_id = conn.execute(max_id_query).scalar() or 0
                sequence_name = f"{CFG.db_table_name}_registro_id_seq"
                conn.execute(text(f"ALTER SEQUENCE {sequence_name} RESTART WITH {max_id + 1}"))

                logging.info(f"IDs reasignados secuencialmente. Máximo ID: {max_id}")
            else:
                logging.debug("Los registro_id ya son secuenciales")

    except Exception as e:
        logging.error(f"Error asegurando IDs secuenciales: {e}")


def export_db_to_csv(engine):
    """Exporta todos los datos de la BD al CSV."""
    if engine is None:
        logging.warning("No hay conexión a BD para exportar.")
        return

    try:
        # Asegurar que los IDs sean secuenciales antes de exportar
        ensure_sequential_ids(engine)

        with engine.connect() as conn:
            # Verificar si la tabla existe y tiene datos
            table_exists_query = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = :table_name
                );
            """)

            table_exists = conn.execute(table_exists_query, {'table_name': CFG.db_table_name}).scalar()

            if not table_exists:
                logging.info("Tabla no existe, no hay datos para exportar.")
                return

            # Contar registros
            count_query = text(f"SELECT COUNT(*) FROM {CFG.db_table_name}")
            count = conn.execute(count_query).scalar()

            if count == 0:
                logging.info("Tabla existe pero no tiene registros.")
                return

            # Exportar datos ordenados por fecha e id
            export_query = text(f"""
                SELECT registro_id, id, fecha, nombre, precio, cantidad, total, total_fact, proveedor
                FROM {CFG.db_table_name}
                ORDER BY registro_id
            """)

            df = pd.read_sql(export_query, conn)

            # Guardar a CSV
            csv_path = Path(CFG.workspace_dir) / CFG.csv_filename
            df.to_csv(csv_path, index=False)
            logging.info(f"Exportados {len(df)} registros desde BD a {CFG.csv_filename}")

    except Exception as e:
        logging.error(f"Error exportando BD a CSV: {e}")


def create_initial_dataframe() -> pd.DataFrame:
    """Crea el DataFrame inicial con datos por defecto."""
    data = {
        'id': [1, 2, 3],
        'fecha': ['2023-01-01', '2023-01-01', '2023-01-01'],
        'nombre': ['Producto inicial 1', 'Producto inicial 2', 'Producto inicial 3'],
        'precio': [100.0, 200.0, 300.0],
        'cantidad': [1.0, 1.0, 1.0],
        'total': [100.0, 200.0, 300.0],
        'total_fact': [600.0, 600.0, 600.0],
        'proveedor': ['Proveedor inicial', 'Proveedor inicial', 'Proveedor inicial']
    }
    return pd.DataFrame(data)


def save_to_csv(df: pd.DataFrame, append: bool = True):
    """Guarda el DataFrame en CSV."""
    csv_path = Path(CFG.workspace_dir) / CFG.csv_filename

    try:
        # Preparar DataFrame para CSV (sin registro_id ya que es generado por BD)
        df_csv = df.copy()

        if append and csv_path.exists():
            # Leer CSV existente y concatenar
            existing_df = pd.read_csv(csv_path)
            combined_df = pd.concat([existing_df, df_csv], ignore_index=True)
            combined_df.to_csv(csv_path, index=False)
            logging.info(f"Agregadas {len(df_csv)} filas al CSV existente")
        else:
            # Crear nuevo CSV
            df_csv.to_csv(csv_path, index=False)
            logging.info(f"Creado nuevo CSV con {len(df_csv)} filas")

    except Exception as e:
        logging.error(f"Error guardando CSV: {e}")
        raise ExtractorError(f"No se pudo guardar en CSV: {e}")


# ---------------------------------------------------------------------------
# Validación de fechas y datos
# ---------------------------------------------------------------------------

def validate_and_get_start_date(session: requests.Session, engine) -> date:
    """Valida la consistencia de datos y determina la fecha de inicio usando BD como fuente de verdad."""
    last_date = get_last_date_from_db(engine)

    if last_date is None:
        # Primera ejecución - crear datos iniciales en BD
        logging.info("Primera ejecución, creando datos iniciales")
        initial_df = create_initial_dataframe()
        save_to_database(initial_df, engine)
        # Exportar a CSV después de guardar en BD
        export_db_to_csv(engine)
        return date(2023, 1, 2)  # Empezar desde el día siguiente

    # Obtener cantidad de facturas de ese día desde la BD
    try:
        with engine.connect() as conn:
            count_query = text(f"""
                SELECT COUNT(*) FROM {CFG.db_table_name}
                WHERE fecha = :target_date
            """)
            num_lineas_db = conn.execute(count_query, {'target_date': last_date}).scalar()
    except Exception as e:
        logging.error(f"Error consultando BD: {e}")
        # Si hay error, asumir que necesitamos revalidar
        return last_date

    # Obtener facturas de ese día desde la API
    lineas_api = fetch_bills_by_date(session, last_date)
    num_lineas_api = len(lineas_api)

    logging.info(f"Fecha {last_date}: BD={num_lineas_db} líneas, API={num_lineas_api} líneas")

    if num_lineas_db == num_lineas_api:
        # Si coinciden, empezar desde el día siguiente
        start_date = last_date + timedelta(days=1)
        logging.info(f"Datos consistentes, empezando desde: {start_date}")
        return start_date
    else:
        # Si no coinciden, limpiar ese día en BD y empezar desde ahí
        logging.info(f"Inconsistencia detectada, limpiando datos del {last_date} en BD")

        try:
            with engine.begin() as conn:
                # Eliminar registros de la fecha inconsistente
                delete_query = text(f"""
                    DELETE FROM {CFG.db_table_name}
                    WHERE fecha = :target_date
                """)
                result = conn.execute(delete_query, {'target_date': last_date})
                deleted_count = result.rowcount
                logging.info(f"Eliminados {deleted_count} registros del {last_date} de la BD")

                # Resetear la secuencia para mantener IDs secuenciales
                # Obtener el máximo ID actual después de la eliminación
                max_id_query = text(f"SELECT COALESCE(MAX(registro_id), 0) FROM {CFG.db_table_name}")
                max_id = conn.execute(max_id_query).scalar()

                # Resetear la secuencia para que el próximo ID sea max_id + 1
                sequence_name = f"{CFG.db_table_name}_registro_id_seq"
                reset_sequence_query = text(f"ALTER SEQUENCE {sequence_name} RESTART WITH {max_id + 1}")
                conn.execute(reset_sequence_query)
                logging.info(f"Secuencia {sequence_name} reseteada a {max_id + 1} para mantener IDs secuenciales")

        except Exception as e:
            logging.error(f"Error eliminando registros o reseteando secuencia: {e}")

        # Re-exportar CSV desde BD actualizada
        export_db_to_csv(engine)

        logging.info(f"Recomenzando desde: {last_date}")
        return last_date


def fetch_bills_by_date(session: requests.Session, target_date: date) -> List[Dict[str, Any]]:
    """Obtiene todas las facturas de una fecha específica desde la API."""
    url = f"{CFG.base_url}?limit=30&order_field=date&type=bill&date={target_date}"
    
    try:
        data = safe_request(session, url)
        
        if not data or not isinstance(data, list):
            return []
        
        # Procesar facturas y extraer items
        lineas = []
        for bill in data:
            if not isinstance(bill, dict):
                continue
                
            purchases = bill.get('purchases', {})
            if not isinstance(purchases, dict) or 'items' not in purchases:
                continue
                
            for item in purchases['items']:
                if not isinstance(item, dict):
                    continue
                    
                provider_name = ""
                if isinstance(bill.get('provider'), dict):
                    provider_name = bill['provider'].get('name', '')
                
                lineas.append({
                    'id': item.get('id'),
                    'fecha': bill.get('date'),
                    'nombre': item.get('name'),
                    'precio': item.get('price'),
                    'cantidad': item.get('quantity'),
                    'total': item.get('total'),
                    'total_fact': bill.get('total'),
                    'proveedor': provider_name
                })
        
        return lineas
        
    except Exception as e:
        logging.error(f"Error obteniendo facturas del {target_date}: {e}")
        return []


# ---------------------------------------------------------------------------
# Descarga y procesamiento de facturas
# ---------------------------------------------------------------------------

def fetch_bills_range(session: requests.Session, start_date: date, end_date: date) -> pd.DataFrame:
    """Descarga facturas en un rango de fechas usando concurrencia."""
    # Generar lista de fechas a procesar
    dates_to_process = []
    current_date = start_date
    while current_date <= end_date:
        dates_to_process.append(current_date)
        current_date += timedelta(days=1)

    if not dates_to_process:
        logging.info("No hay fechas para procesar")
        return pd.DataFrame()

    logging.info(f"Procesando {len(dates_to_process)} fechas concurrentemente con {CFG.concurrent_requests} hilos")

    # Ejecutar extracción concurrente
    async def async_fetch():
        return await fetch_bills_concurrent(dates_to_process, CFG.concurrent_requests)

    concurrent_results = asyncio.run(async_fetch())

    # Procesar resultados de cada fecha
    all_bills = []
    for target_date, bills_data in concurrent_results.items():
        if bills_data:
            processed_bills = process_bills_data_async(bills_data, target_date)
            all_bills.extend(processed_bills)
            logging.info(f"Fecha {target_date}: {len(processed_bills)} líneas procesadas")
        else:
            logging.info(f"Fecha {target_date}: No hay facturas")

    if not all_bills:
        logging.info("No se encontraron facturas en el rango especificado")
        return pd.DataFrame()

    # Convertir a DataFrame y limpiar datos
    df = pd.DataFrame(all_bills)
    return clean_bills_data(df)


def clean_bills_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y valida los datos de facturas."""
    if df.empty:
        return df
    
    try:
        # Convertir tipos de datos
        numeric_columns = ['precio', 'cantidad', 'total', 'total_fact']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        # Limpiar strings
        string_columns = ['nombre', 'proveedor']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # Validar fechas
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
            df = df.dropna(subset=['fecha'])
        
        # Eliminar filas con datos críticos faltantes
        df = df.dropna(subset=['id', 'nombre'])
        
        logging.info(f"Datos limpiados: {len(df)} registros válidos")
        return df
        
    except Exception as e:
        logging.error(f"Error limpiando datos: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Base de datos PostgreSQL
# ---------------------------------------------------------------------------

def get_database_engine():
    """Crea el engine de SQLAlchemy para PostgreSQL."""
    if not SQLALCHEMY_AVAILABLE:
        return None
    
    db_url = os.getenv(CFG.db_url_env)
    if not db_url:
        logging.warning(f"Variable {CFG.db_url_env} no encontrada. Solo se guardará en CSV.")
        return None
    
    try:
        engine = create_engine(db_url)
        # Probar conexión
        with engine.connect():
            pass
        logging.info("Conexión a PostgreSQL establecida")
        return engine
    except Exception as e:
        logging.error(f"Error conectando a PostgreSQL: {e}")
        return None


def save_to_database(df: pd.DataFrame, engine):
    """Guarda el DataFrame en PostgreSQL con tipos de datos optimizados y manejo de reconexión."""
    if engine is None or df.empty:
        return

    # Mapeo de tipos de columna para optimización
    dtype_mapping = {
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

    max_db_retries = 3
    for attempt in range(1, max_db_retries + 1):
        try:
            with engine.begin() as conn:  # Usar begin() para manejar transacciones automáticamente
                # Primero verificar si la tabla existe y tiene la estructura correcta
                table_exists_query = text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = :table_name
                    );
                """)

                table_exists = conn.execute(table_exists_query, {'table_name': CFG.db_table_name}).scalar()

                if not table_exists:
                    logging.info(f"Creando tabla {CFG.db_table_name}...")
                    create_table_sql = text(f"""
                        CREATE TABLE {CFG.db_table_name} (
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
                    """)
                    conn.execute(create_table_sql)
                    logging.info(f"Tabla {CFG.db_table_name} creada exitosamente")

                # Convertir fecha para la base de datos
                df_db = df.copy()
                if 'fecha' in df_db.columns:
                    df_db['fecha'] = pd.to_datetime(df_db['fecha'], errors='coerce').dt.date

                # Insertar datos usando to_sql con transacción
                df_db.to_sql(
                    CFG.db_table_name,
                    conn,
                    if_exists="append",
                    index=False,
                    dtype=dtype_mapping
                )
                logging.info(f"Guardadas {len(df_db)} facturas en PostgreSQL")
                return  # Éxito, salir de la función

        except Exception as e:
            logging.error(f"Error guardando en PostgreSQL (intento {attempt}/{max_db_retries}): {e}")
            logging.error(f"Tipo de error: {type(e).__name__}")

            if attempt < max_db_retries:
                wait_time = 5 * attempt  # Esperar 5, 10, 15 segundos
                logging.info(f"Reintentando en {wait_time} segundos...")
                time.sleep(wait_time)

                # Intentar reconectar el engine
                try:
                    engine.dispose()  # Cerrar conexiones existentes
                    db_url = os.getenv(CFG.db_url_env)
                    if db_url:
                        from sqlalchemy import create_engine
                        engine = create_engine(db_url)
                        logging.info("Engine reconectado exitosamente")
                    else:
                        logging.error("No se puede reconectar: DATABASE_URL no encontrada")
                        break
                except Exception as reconn_error:
                    logging.error(f"Error reconectando engine: {reconn_error}")
            else:
                logging.error(f"Fallaron todos los {max_db_retries} intentos de guardar en PostgreSQL")
                raise  # Re-lanzar la excepción después de todos los intentos


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def main():
    """Función principal del extractor de facturas de proveedores."""
    setup_logging(CFG.log_level)
    logging.info("Iniciando extractor de facturas de proveedores optimizado")
    
    try:
        # Configurar sesión HTTP
        api_key = get_api_key()
        session = create_session(api_key)
        
        # Configurar conexión a base de datos (opcional)
        engine = get_database_engine()
        
        # Validar datos existentes y determinar fecha de inicio
        start_date = validate_and_get_start_date(session, engine)
        end_date = date.today()

        if start_date > end_date:
            logging.info("No hay fechas nuevas para procesar")
            # Aún así exportar CSV desde BD existente
            export_db_to_csv(engine)
            return

        logging.info(f"Procesando facturas desde {start_date} hasta {end_date}")

        # Descargar y procesar facturas
        df = fetch_bills_range(session, start_date, end_date)

        if df.empty:
            logging.info("No se encontraron facturas nuevas para procesar")
            # Aún así exportar CSV desde BD existente
            export_db_to_csv(engine)
            return

        # Guardar PRIMERO en PostgreSQL (fuente de verdad)
        if engine:
            save_to_database(df, engine)

        # Luego exportar CSV desde BD actualizada
        if CFG.require_csv:
            export_db_to_csv(engine)

        logging.info(f"Proceso completado exitosamente. Procesadas {len(df)} líneas de facturas.")
        
    except ExtractorError as e:
        logging.error(f"Error del extractor: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error inesperado: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_dotenv()
    main() 