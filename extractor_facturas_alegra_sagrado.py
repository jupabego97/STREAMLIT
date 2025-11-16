"""
Extractor de facturas de Alegra API a PostgreSQL
Optimizado para concurrencia asíncrona y uso de la base de datos como fuente de verdad
"""

import os
import requests
import pandas as pd
import datetime
import logging
from sqlalchemy import create_engine, types as sa_types, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import sys
import asyncio
import aiohttp
import nest_asyncio

# Cargar variables de entorno desde .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# -----------------------------
# Configuración de logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Variable para controlar exportación a CSV al final
EXPORT_TO_CSV = True  # Cambiar a True para exportar

# -----------------------------
# Variables de configuración
# -----------------------------
API_KEY = "bmFub3Ryb25pY3NhbHNvbmRlbGF0ZWNub2xvZ2lhQGdtYWlsLmNvbTphMmM4OTA3YjE1M2VmYTc0ODE5ZA=="  # Reemplaza con tu clave real
HEADERS = {
    "accept": "application/json",
    "authorization": f"Basic {API_KEY}"
}
LIMIT = 30
CONCURRENT_REQUESTS = 7    # Máximo de peticiones simultáneas
RETRY_DELAY_429 = 60         # Segundos a esperar tras error 429
NETWORK_ERROR_DELAY = 5      # Segundos a esperar tras excepción de red
MAX_RETRIES = 5              # Número máximo de reintentos por página


# -----------------------------
# Funciones asíncronas para extracción concurrente
# -----------------------------
async def fetch_invoice_batch(session, start, batch_size=LIMIT):
    """
    Extrae una página de facturas y maneja errores 429 con reintento.
    """
    url = (
        f"https://api.alegra.com/api/v1/invoices"
        f"?start={start}&order_direction=ASC&order_field=id&limit={batch_size}"
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.get(url, headers=HEADERS, timeout=30) as response:
                status = response.status
                if status == 200:
                    data = await response.json()
                    logger.info(f"✅ Página start={start} extraída con {len(data)} facturas.")
                    return pd.DataFrame(data)
                elif status == 429:
                    logger.warning(
                        f"⚠️ Error 429 en start={start}. "
                        f"Esperando {RETRY_DELAY_429}s antes de reintentar... "
                        f"(Intento {attempt}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(RETRY_DELAY_429)
                else:
                    logger.error(f"❌ Error {status} en start={start}. No se reintentará.")
                    return pd.DataFrame()
        except Exception as e:
            logger.error(
                f"💥 Excepción en start={start}: {e}. "
                f"Esperando {NETWORK_ERROR_DELAY}s antes de reintentar... "
                f"(Intento {attempt}/{MAX_RETRIES})"
            )
            await asyncio.sleep(NETWORK_ERROR_DELAY)
    logger.error(f"⛔ Fallo definitivo en start={start} tras {MAX_RETRIES} intentos.")
    return pd.DataFrame()


async def extract_invoices_concurrent(start_id, end_id, batch_size=LIMIT, concurrency=CONCURRENT_REQUESTS):
    """
    Extrae facturas de Alegra API en paralelo usando asyncio.
    """
    nest_asyncio.apply()
    semaphore = asyncio.Semaphore(concurrency)
    all_dfs = []

    async with aiohttp.ClientSession() as session:
        async def bounded_fetch(start):
            async with semaphore:
                return await fetch_invoice_batch(session, start, batch_size)

        starts = list(range(start_id, end_id + 1, batch_size))
        tasks = [bounded_fetch(start) for start in starts]
        results = await asyncio.gather(*tasks)

    # Concatenar todos los DataFrames no vacíos
    for df in results:
        if not df.empty:
            all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        return combined
    else:
        return pd.DataFrame()


# -----------------------------
# Clase principal del extractor
# -----------------------------
class AlegraFacturasExtractor:
    """
    Extracción de facturas de Alegra API a PostgreSQL,
    optimizado para concurrencia asíncrona y uso de la base de datos como fuente de verdad.
    """
    def __init__(self):
        self.engine = None
        self.headers = HEADERS
        self.dtype_mapping = {
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

    def connect_database(self):
        """Conectar a la base de datos PostgreSQL usando variables de entorno."""
        try:
            db_url = os.getenv('DATABASE_URL')

            if db_url:
                connection_string = db_url
            else:
                db_user = os.getenv('DB_USER', 'postgres')
                db_password = os.getenv('DB_PASSWORD', 'tUATSaWIdtPddGNSPhUcRWKnrdOTqbKX')
                db_host = os.getenv('DB_HOST', 'mainline.proxy.rlwy.net')
                db_port = os.getenv('DB_PORT', '42296')
                db_name = os.getenv('DB_NAME', 'railway')
                connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

            self.engine = create_engine(connection_string)

            # Probar conexión
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            logger.info("Conexión a la base de datos establecida exitosamente")
            return True

        except Exception as e:
            logger.error(f"Error conectando a la base de datos: {e}")
            return False

    def create_table_if_not_exists(self):
        """Crear tabla facturas si no existe o verificar estructura."""
        if not self.engine:
            logger.error("No hay conexión a la base de datos")
            return False

        try:
            check_table_sql = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'facturas'
            );
            """
            with self.engine.connect() as conn:
                table_exists = conn.execute(text(check_table_sql)).fetchone()
                if table_exists:
                    table_exists = table_exists[0]
                else:
                    table_exists = False

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
                    logger.info("Tabla facturas creada exitosamente")
                else:
                    # Verificar si la columna item_id existe, si no, agregarla
                    check_column_sql = """
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns
                        WHERE table_schema = 'public'
                        AND table_name = 'facturas'
                        AND column_name = 'item_id'
                    );
                    """
                    column_exists = conn.execute(text(check_column_sql)).fetchone()
                    if not column_exists or not column_exists[0]:
                        logger.info("Agregando columna item_id a tabla existente...")
                        add_column_sql = """
                        ALTER TABLE facturas ADD COLUMN item_id INTEGER DEFAULT 0 NOT NULL;
                        """
                        conn.execute(text(add_column_sql))
                        conn.commit()
                        logger.info("Columna item_id agregada exitosamente")

                    # Verificar si existen los índices básicos
                    create_indexes_sql = """
                    CREATE INDEX IF NOT EXISTS idx_facturas_id ON facturas(id);
                    CREATE INDEX IF NOT EXISTS idx_facturas_fecha ON facturas(fecha);
                    CREATE INDEX IF NOT EXISTS idx_facturas_item_id ON facturas(item_id);
                    """
                    conn.execute(text(create_indexes_sql))
                    conn.commit()
                    logger.info("Tabla facturas existe - índices verificados")

            return True

        except Exception as e:
            logger.error(f"Error creando/verificando tabla: {e}")
            return False

    def get_last_invoice_id(self):
        """Obtener el ID de la última factura procesada desde la BD."""
        if not self.engine:
            return None

        try:
            query = "SELECT MAX(id) as max_id FROM facturas"
            with self.engine.connect() as conn:
                result = conn.execute(text(query)).fetchone()
            if result and result.max_id:
                logger.info(f"Última factura en BD: {result.max_id}")
                return result.max_id
            else:
                logger.info("No hay facturas previas en la BD, iniciando desde fecha predeterminada")
                return None

        except Exception as e:
            logger.error(f"Error obteniendo última factura: {e}")
            return None

    def get_starting_invoice_id(self):
        """Determinar el ID de factura desde donde iniciar la extracción."""
        last_id = self.get_last_invoice_id()
        if last_id:
            return last_id + 1
        else:
            logger.info("No hay facturas previas en la BD, iniciando desde ID 1 (modo pruebas)")
            return 1

    def check_table_exists(self):
        """Verificar si la tabla facturas existe y tiene datos."""
        if not self.engine:
            return False

        try:
            with self.engine.connect() as conn:
                # Verificar si la tabla existe
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = 'facturas'
                    );
                """)).fetchone()

                if not result or not result[0]:
                    return False

                # Verificar si la tabla tiene datos
                count_result = conn.execute(text("SELECT COUNT(*) FROM facturas")).fetchone()
                return count_result and count_result[0] > 0

        except Exception as e:
            logger.error(f"Error verificando tabla: {e}")
            return False

    def get_latest_invoice_id(self):
        """Obtener el ID de la factura más reciente de la API."""
        try:
            current_date = datetime.datetime.now()
            search_date = current_date - datetime.timedelta(days=1)

            if not self.engine or not self.check_table_exists():
                current_date = datetime.datetime(2022, 11, 1)
                search_date = current_date + datetime.timedelta(days=30)
                logger.info("Usando fecha inicial predeterminada: 2022-11-01")

            url = (
                f"https://api.alegra.com/api/v1/invoices"
                f"?date_beforeOrNow={search_date.strftime('%Y-%m-%d')}"
                f"&order_direction=DESC&limit=1"
            )
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            if data and len(data) > 0:
                latest_id = int(data[0]['id'])
                logger.info(f"Última factura en API: {latest_id}")
                return latest_id
            else:
                logger.warning("No se encontraron facturas recientes en la API")
                return None

        except Exception as e:
            logger.error(f"Error obteniendo última factura de API: {e}")
            return None

    def extract_invoices_batch(self, start_id, end_id, batch_size=LIMIT):
        """
        Extraer facturas concurrentemente desde la API de Alegra.
        Reemplaza la versión síncrona original.
        """
        try:
            async def async_extract():
                return await extract_invoices_concurrent(
                    start_id=start_id,
                    end_id=end_id,
                    batch_size=batch_size,
                    concurrency=CONCURRENT_REQUESTS
                )

            df = asyncio.run(async_extract())
            logger.info(f"Total de facturas extraídas concurrentemente: {len(df)}")
            return df

        except Exception as e:
            logger.error(f"Error en extracción concurrente: {e}")
            return pd.DataFrame()

    def clean_invoice_data(self, df):
        """Limpiar y procesar los datos de facturas."""
        if df.empty:
            return df

        logger.info("Procesando datos de facturas...")

        columns_to_drop = [
            'observations', 'payments', 'subtotal', 'barCodeContent', 'total',
            'numberTemplate', 'dueDate', 'stamp', 'warehouse', 'term', 'anotation',
            'termsConditions', 'status', 'priceList', 'costCenter', 'paymentForm',
            'type', 'discount', 'tax', 'balance', 'decimalPrecision', 'operationType',
            'printingTemplate', 'station', 'retentions'
        ]
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns=existing_columns_to_drop, errors='ignore')

        def extract_names(row):
            try:
                client_val = row['client']
                if isinstance(client_val, dict) and 'name' in client_val:
                    row['client'] = client_val['name']

                seller_val = row['seller']
                if isinstance(seller_val, dict) and 'name' in seller_val:
                    row['seller'] = seller_val['name']
                elif not seller_val:
                    row['seller'] = 'No se ha registrado un vendedor'
            except (KeyError, TypeError):
                pass
            return row

        df = df.apply(extract_names, axis=1)

        def clean_items(row):
            try:
                items_val = row['items']
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

    def transform_to_line_items(self, df):
        """Transformar facturas a líneas de items individuales."""
        if df.empty:
            return pd.DataFrame()

        logger.info("Transformando a líneas de items individuales...")
        line_items = []

        for _, row in df.iterrows():
            try:
                items_val = row['items']
                if isinstance(items_val, list):
                    for item in items_val:
                        # Asegurar que metodo_val nunca sea None para la BD
                        payment_method_from_api = row.get('paymentMethod')
                        metodo_val = payment_method_from_api if payment_method_from_api and pd.notna(payment_method_from_api) else 'Sin especificar'
                        
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
                            'totalfact': float(row.get('totalPaid', 0)) if row.get('totalPaid') is not None else 0.0,
                            'metodo': metodo_val,
                            'vendedor': vendedor_val
                        }
                        line_items.append(line_item)
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Error procesando fila con ID {row.get('id', 'desconocido')}: {e}")
                continue

        result_df = pd.DataFrame(line_items)
        logger.info(f"Generadas {len(result_df)} líneas de items")
        return result_df

    def insert_to_database(self, df):
        """Insertar datos en la base de datos."""
        if df.empty:
            logger.info("No hay datos para insertar")
            return True

        try:
            logger.info(f"Insertando {len(df)} registros en la base de datos...")
            df.to_sql(
                "facturas",
                self.engine,
                if_exists="append",
                index=False,
                dtype=self.dtype_mapping
            )
            logger.info("Datos insertados exitosamente")
            return True

        except Exception as e:
            logger.error(f"Error insertando datos: {e}")
            return False

    def export_to_csv(self, filename="facturas_backup.csv"):
        """Exportar todos los datos de la BD a CSV."""
        if not self.engine:
            return

        try:
            logger.info(f"Exportando datos a {filename}...")
            query = "SELECT * FROM facturas ORDER BY id, indx"
            df = pd.read_sql(query, self.engine)
            df_export = df.drop(columns=['indx', 'created_at'], errors='ignore')
            df_export.to_csv(filename, index=False)
            logger.info(f"Datos exportados exitosamente a {filename} ({len(df_export)} registros)")

        except Exception as e:
            logger.error(f"Error exportando a CSV: {e}")

    def run_extraction(self):
        """Ejecutar el proceso completo de extracción."""
        logger.info("=== Iniciando extracción de facturas Alegra ===")

        if not self.connect_database():
            logger.error("No se pudo conectar a la base de datos")
            return False

        if not self.create_table_if_not_exists():
            logger.error("No se pudo crear/verificar la tabla")
            return False

        start_id = self.get_starting_invoice_id()
        if not start_id:
            logger.error("No se pudo determinar el ID de inicio")
            return False

        end_id = self.get_latest_invoice_id()
        if not end_id:
            logger.error("No se pudo determinar el ID final")
            return False

        if start_id > end_id:
            logger.info("No hay nuevas facturas para procesar")
            self.export_to_csv()
            return True

        raw_df = self.extract_invoices_batch(start_id, end_id)
        if raw_df.empty:
            logger.info("No se extrajeron nuevas facturas")
            self.export_to_csv()
            return True

        cleaned_df = self.clean_invoice_data(raw_df)
        if cleaned_df.empty:
            logger.error("Error procesando datos de facturas")
            return False

        line_items_df = self.transform_to_line_items(cleaned_df)
        if line_items_df.empty:
            logger.error("Error transformando a líneas de items")
            return False

        if not self.insert_to_database(line_items_df):
            logger.error("Error insertando datos en la base de datos")
            return False

        self.export_to_csv()
        logger.info("=== Extracción completada exitosamente ===")
        return True


def main():
    """Función principal."""
    extractor = AlegraFacturasExtractor()

    try:
        success = extractor.run_extraction()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("Proceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
