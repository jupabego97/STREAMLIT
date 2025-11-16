#!/usr/bin/env python3
"""
Extractor concurrente de items desde la API de Alegra
----------------------------------------------------

▶ Funciones principales (sin cambios funcionales):
   1. Extrae items de productos desde la API de Alegra
   2. Procesa y limpia los datos con campos personalizados
   3. Guarda en PostgreSQL (y opcionalmente en CSV)
   4. Añade 3 ítems “SERVICIO TÉCNICO” iniciales
   5. Genera un CSV al final como respaldo

▶ Cambio clave:
   · La función de descarga ahora usa paginación **asíncrona y concurrente**  
     (hasta N peticiones simultáneas) con manejo de errores 429 y reintentos.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import nest_asyncio
import pandas as pd
import requests
from dotenv import load_dotenv

# Dependencias opcionales para PostgreSQL
try:
    from sqlalchemy import create_engine, types as sa_types, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logging.warning("SQLAlchemy no está disponible. Solo se guardará en CSV.")


# ───────────────────────────────────────────────────────────────────────────────
# Configuración global
# ───────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class Config:
    # API
    base_url: str = "https://api.alegra.com/api/v1/items"
    metadata_url: str = "https://api.alegra.com/api/v1/items?metadata=true"
    api_key_env: str = "ALEGRA_API_KEY"
    api_key_default: str = (
        "bmFub3Ryb25pY3NhbHNvbmRlbGF0ZWNub2xvZ2lhQGdtYWlsLmNvbTphMmM4OTA3YjE1M2VmYTc0ODE5ZA=="
    )

    # Descarga
    page_size: int = 30
    timeout_seconds: int = 30
    max_retries: int = 3
    backoff_factor: float = 1.5

    # Concurrencia
    concurrent_requests: int = 10          # Máximo de requests simultáneas
    retry_delay_429: int = 60              # Espera (s) tras 429
    network_error_delay: int = 5           # Espera (s) tras fallo de red

    # Ficheros
    csv_filename: str = "items.csv"
    workspace_dir: str = "."

    # Base de datos
    db_url_env: str = "DATABASE_URL"
    db_table_name: str = "items"

    # Otros
    require_csv: bool = False
    log_level: int = logging.INFO


CFG = Config()


# ───────────────────────────────────────────────────────────────────────────────
# Utilidades generales
# ───────────────────────────────────────────────────────────────────────────────
def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class ExtractorError(Exception):
    """Excepción personalizada para el extractor."""


def get_api_key() -> str:
    key = os.getenv(CFG.api_key_env, CFG.api_key_default)
    if key == CFG.api_key_default:
        logging.warning(f"Usando API-Key por defecto: variable {CFG.api_key_env} no encontrada.")
    return key


def create_sync_session(api_key: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"accept": "application/json", "authorization": f"Basic {api_key}"})
    return s


# ───────────────────────────────────────────────────────────────────────────────
# Descarga concurrente de items (NEW)
# ───────────────────────────────────────────────────────────────────────────────
async def fetch_item_page(
    session, start: int, limit: int
) -> List[Dict[str, Any]]:
    """
    Descarga una página de items. Maneja 429 y reintenta.
    Devuelve una lista (posiblemente vacía) de dicts.
    """
    url = f"{CFG.base_url}?start={start}&limit={limit}&order_field=id"
    for attempt in range(1, CFG.max_retries + 1):
        try:
            async with session.get(url, timeout=CFG.timeout_seconds) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logging.info(f"✅ start={start} → {len(data)} items")
                    return data
                if resp.status == 429:
                    logging.warning(
                        f"⚠️ 429 en start={start}. "
                        f"Esperando {CFG.retry_delay_429}s (intento {attempt}/{CFG.max_retries})"
                    )
                    await asyncio.sleep(CFG.retry_delay_429)
                else:
                    logging.error(f"❌ Error {resp.status} en start={start}.")
                    return []
        except Exception as e:
            logging.error(
                f"💥 Excepción en start={start}: {e}. "
                f"Reintentando en {CFG.network_error_delay}s (intento {attempt}/{CFG.max_retries})"
            )
            await asyncio.sleep(CFG.network_error_delay)
    logging.error(f"⛔ Fallo definitivo en start={start} tras {CFG.max_retries} intentos.")
    return []


async def fetch_all_items_concurrent(total_items: int, api_key: str) -> List[Dict[str, Any]]:
    """Descarga todos los items en paralelo respetando CFG.concurrent_requests."""
    nest_asyncio.apply()
    sem = asyncio.Semaphore(CFG.concurrent_requests)
    all_pages: List[List[Dict[str, Any]]] = []

    import aiohttp

    async with aiohttp.ClientSession(
        headers={"accept": "application/json", "authorization": f"Basic {api_key}"}
    ) as session:

        async def bounded_fetch(offset):
            async with sem:
                return await fetch_item_page(session, offset, CFG.page_size)

        starts = list(range(0, total_items, CFG.page_size))
        tasks = [bounded_fetch(offset) for offset in starts]
        results = await asyncio.gather(*tasks)

    for page in results:
        if page:
            all_pages.extend(page)
    return all_pages


# ───────────────────────────────────────────────────────────────────────────────
# Procesamiento (sin cambios relevantes)
# ───────────────────────────────────────────────────────────────────────────────
def extract_custom_field(custom_fields: List[Dict[str, Any]], field_name: str) -> Optional[str]:
    if not isinstance(custom_fields, list):
        return None
    for f in custom_fields:
        if f.get("name") == field_name:
            return f.get("value")
    return None


def extract_price(price_list: List[Dict[str, Any]]) -> Optional[float]:
    if price_list and isinstance(price_list, list):
        try:
            return float(price_list[0].get("price", 0))
        except Exception:
            return None
    return None


def extract_inventory_info(inv: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    if not isinstance(inv, dict):
        return None, None
    fecha = inv.get("initialQuantityDate")
    qty = inv.get("availableQuantity")
    try:
        qty = float(qty) if qty is not None else None
    except Exception:
        qty = None
    return fecha, qty


def clean_items_data(items: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in items:
        try:
            custom = item.get("customFields", [])
            inv = item.get("inventory", {})
            rows.append(
                {
                    "id": int(item.get("id", 0)),
                    "nombre": item.get("name"),
                    "codigo_barras": extract_custom_field(custom, "Código de barras"),
                    "familia": extract_custom_field(custom, "FAMILIA"),
                    "precio": extract_price(item.get("price", [])),
                    "fecha_inicial": extract_inventory_info(inv)[0],
                    "cantidad_disponible": extract_inventory_info(inv)[1],
                }
            )
        except Exception as e:
            logging.warning(f"Item {item.get('id')} con error: {e}")
    df = pd.DataFrame(rows)
    logging.info(f"Procesados {len(df)} items.")
    return df


# ───────────────────────────────────────────────────────────────────────────────
# Base de datos & helpers (igual que tu versión anterior)
# ───────────────────────────────────────────────────────────────────────────────
def get_database_engine():
    if not SQLALCHEMY_AVAILABLE:
        raise ExtractorError("SQLAlchemy no instalado.")
    db_url = os.getenv(CFG.db_url_env)
    if not db_url:
        raise ExtractorError(f"Variable {CFG.db_url_env} no configurada.")
    try:
        engine = create_engine(db_url)
        with engine.connect():
            pass
        logging.info("Conexión a PostgreSQL OK.")
        return engine
    except Exception as e:
        raise ExtractorError(f"No se pudo conectar a PostgreSQL: {e}")


def create_initial_dataframe() -> pd.DataFrame:
    data = {
        "id": [1, 2, 3],
        "nombre": [
            "SERVICIO TECNICO",
            "SERVICIO TECNICO CONSOLA",
            "SERVICIO TECNICO IMPRESORA",
        ],
        "codigo_barras": ["0491", "0492", "0493"],
        "familia": ["SERVICIOS", "SERVICIOS", "SERVICIOS"],
        "precio": [0.0, 0.0, 0.0],
        "fecha_inicial": ["2023-01-02"] * 3,
        "cantidad_disponible": [0.0, 0.0, 0.0],
    }
    return pd.DataFrame(data)


def save_to_database(df: pd.DataFrame, engine, mode: Literal["replace", "append"]):
    if df.empty:
        logging.info("DataFrame vacío. Nada que insertar.")
        return
    dtype_map = {
        "id": sa_types.INTEGER(),
        "nombre": sa_types.String(300),
        "codigo_barras": sa_types.String(50),
        "familia": sa_types.String(100),
        "precio": sa_types.NUMERIC(12, 2),
        "fecha_inicial": sa_types.DATE(),
        "cantidad_disponible": sa_types.NUMERIC(10, 2),
    }
    df.to_sql(
        CFG.db_table_name,
        engine,
        if_exists=mode,
        index=False,
        dtype=dtype_map,
        method="multi",
    )
    logging.info(f"Guardados {len(df)} items en PostgreSQL ({mode}).")


def load_entire_database(engine) -> pd.DataFrame:
    with engine.connect() as con:
        exists = con.execute(
            text(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = :t)",
            ),
            {"t": CFG.db_table_name},
        ).scalar()
        if not exists:
            return pd.DataFrame()
    return pd.read_sql_table(CFG.db_table_name, engine)


def save_dataframe_to_csv(df: pd.DataFrame):
    path = Path(CFG.workspace_dir) / CFG.csv_filename
    df.to_csv(path, index=False)
    logging.info(f"CSV generado: {path} ({len(df)} rows)")


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────
def main():
    setup_logging(CFG.log_level)
    load_dotenv()
    logging.info(">> Extractor concurrente de items (Alegra) <<")

    api_key = get_api_key()
    sync_session = create_sync_session(api_key)

    # 1. Total de items
    meta = sync_session.get(CFG.metadata_url, timeout=CFG.timeout_seconds).json()
    total_items = int(meta["metadata"]["total"])
    logging.info(f"Total items reportados por la API: {total_items}")

    # 2. Conexión BD
    engine = get_database_engine()

    # 3. Tabla con 3 ítems base
    save_to_database(create_initial_dataframe(), engine, "replace")

    # 4. Descarga concurrente
    logging.info("Descargando items de la API en paralelo...")
    all_items = asyncio.run(fetch_all_items_concurrent(total_items, api_key))
    if not all_items:
        logging.warning("No se obtuvieron items desde la API.")
    else:
        df_items = clean_items_data(all_items)
        save_to_database(df_items, engine, "append")

    # 5. CSV de respaldo
    if CFG.require_csv:
        df_total = load_entire_database(engine)
        save_dataframe_to_csv(df_total)

    logging.info("Proceso finalizado OK.")


if __name__ == "__main__":
    main()
