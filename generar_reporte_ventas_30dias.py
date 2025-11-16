#!/usr/bin/env python3
"""
Generador de Reportes de Ventas de Últimos 30 Días
---------------------------------------------------

▶ **Funcionalidades principales:**
   1. Genera tabla de reportes de facturas de ventas de los últimos 30 días
   2. Incluye información de familia desde items
   3. Calcula precio promedio de compra desde facturas_proveedor (últimas 3 compras)
   4. Calcula proveedor moda desde facturas_proveedor (últimas 3 compras)
   5. Reemplaza tabla completa cada ejecución (TRUNCATE + INSERT)

▶ **Datos extraídos:**
   - De facturas: nombre, precio, cantidad, metodo, vendedor
   - De items: familia (relacionado por item_id)
   - De facturas_proveedor: precio_promedio_compra, proveedor_moda (calculados)

Requisitos:
```bash
pip install pandas sqlalchemy psycopg2-binary python-dotenv
```
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

# Importaciones para PostgreSQL
try:
    from sqlalchemy import create_engine, types as sa_types, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    logging.warning("SQLAlchemy no está disponible.")


# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------

# Database settings
DB_URL_ENV = "DATABASE_URL"
TABLE_NAME = "reportes_ventas_30dias"
LOG_LEVEL = logging.INFO


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


class ReporteError(Exception):
    """Excepción personalizada para errores del generador de reportes."""
    pass


def get_database_engine():
    """Crea el engine de SQLAlchemy para PostgreSQL."""
    if not SQLALCHEMY_AVAILABLE:
        raise ReporteError("SQLAlchemy no está disponible. Instala: pip install sqlalchemy")
    
    db_url = os.getenv(DB_URL_ENV)
    if not db_url:
        raise ReporteError(f"Variable {DB_URL_ENV} no encontrada. Configura la URL de la base de datos.")
    
    try:
        engine = create_engine(db_url)
        # Probar conexión
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logging.info("Conexión a PostgreSQL establecida")
        return engine
    except Exception as e:
        raise ReporteError(f"No se pudo conectar a PostgreSQL: {e}")


def create_report_table(engine):
    """Crea la tabla de reportes de ventas de 30 días si no existe."""
    if engine is None:
        raise ReporteError("No hay conexión a la base de datos")
    
    try:
        with engine.begin() as conn:
            # Verificar si la tabla existe
            check_table_sql = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = :table_name
                );
            """)
            table_exists = conn.execute(check_table_sql, {'table_name': TABLE_NAME}).scalar()
            
            if not table_exists:
                # Crear tabla nueva
                create_table_sql = text(f"""
                    CREATE TABLE {TABLE_NAME} (
                        id SERIAL PRIMARY KEY,
                        nombre VARCHAR(500) NOT NULL,
                        precio DECIMAL(12,2) NOT NULL,
                        cantidad INTEGER NOT NULL,
                        metodo VARCHAR(50) NOT NULL,
                        vendedor VARCHAR(100) NOT NULL,
                        familia VARCHAR(100),
                        precio_promedio_compra DECIMAL(12,2),
                        proveedor_moda VARCHAR(300),
                        fecha_venta DATE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE INDEX idx_{TABLE_NAME}_nombre ON {TABLE_NAME}(nombre);
                    CREATE INDEX idx_{TABLE_NAME}_fecha_venta ON {TABLE_NAME}(fecha_venta);
                """)
                conn.execute(create_table_sql)
                logging.info(f"Tabla {TABLE_NAME} creada exitosamente")
            else:
                logging.info(f"Tabla {TABLE_NAME} ya existe")
            return True
            
    except Exception as e:
        logging.error(f"Error creando/verificando tabla {TABLE_NAME}: {e}")
        raise ReporteError(f"No se pudo crear/verificar la tabla: {e}")


def generate_report(engine):
    """Genera el reporte de ventas de los últimos 30 días con métricas de proveedor."""
    if engine is None:
        raise ReporteError("No hay conexión a la base de datos")
    
    # Calcular fecha límite (hoy - 30 días)
    fecha_limite = date.today() - timedelta(days=30)
    logging.info(f"Generando reporte desde {fecha_limite} hasta {date.today()}")
    
    try:
        # Ejecutar query y obtener resultados (fuera de transacción)
        query_sql = text("""
            WITH facturas_30dias AS (
                -- Facturas de últimos 30 días con JOIN a items para obtener familia
                SELECT 
                    f.nombre,
                    f.precio,
                    f.cantidad,
                    f.metodo,
                    f.vendedor,
                    f.fecha AS fecha_venta,
                    i.familia
                FROM facturas f
                LEFT JOIN items i ON f.item_id = i.id
                WHERE f.fecha >= :fecha_limite
            ),
            nombres_unicos AS (
                -- Obtener nombres únicos para calcular métricas una sola vez por producto
                SELECT DISTINCT nombre
                FROM facturas_30dias
            ),
            ultimas_compras AS (
                -- Para cada nombre único, obtener las últimas 3 compras de los últimos 3 meses
                SELECT 
                    nu.nombre,
                    fp.precio,
                    fp.proveedor,
                    fp.fecha,
                    ROW_NUMBER() OVER (PARTITION BY nu.nombre ORDER BY fp.fecha DESC) as rn
                FROM nombres_unicos nu
                LEFT JOIN facturas_proveedor fp ON nu.nombre = fp.nombre
                    AND fp.fecha >= CURRENT_DATE - INTERVAL '3 months'
            ),
            compras_relevantes AS (
                -- Filtrar solo las últimas 3 compras (todas dentro de los últimos 3 meses)
                SELECT nombre, precio, proveedor
                FROM ultimas_compras
                WHERE rn <= 3
            ),
            proveedor_conteo AS (
                -- Contar frecuencia de cada proveedor por producto
                SELECT 
                    nombre,
                    proveedor,
                    COUNT(*) as frecuencia
                FROM compras_relevantes
                WHERE proveedor IS NOT NULL
                GROUP BY nombre, proveedor
            ),
            proveedor_ranking AS (
                -- Obtener proveedor más frecuente (moda) por producto
                SELECT DISTINCT ON (nombre)
                    nombre,
                    proveedor AS proveedor_moda
                FROM proveedor_conteo
                ORDER BY nombre, frecuencia DESC, proveedor
            ),
            metricas_proveedor AS (
                -- Calcular promedio de precio y obtener moda de proveedor
                SELECT 
                    mp.nombre,
                    mp.precio_promedio_compra,
                    pr.proveedor_moda
                FROM (
                    -- Promedio de precio
                    SELECT 
                        nombre,
                        AVG(precio) AS precio_promedio_compra
                    FROM compras_relevantes
                    GROUP BY nombre
                ) mp
                LEFT JOIN proveedor_ranking pr ON mp.nombre = pr.nombre
            )
            -- Unir facturas con métricas calculadas
            SELECT 
                f30.nombre,
                f30.precio,
                f30.cantidad,
                f30.metodo,
                f30.vendedor,
                f30.familia,
                mp.precio_promedio_compra,
                mp.proveedor_moda,
                f30.fecha_venta
            FROM facturas_30dias f30
            LEFT JOIN metricas_proveedor mp ON f30.nombre = mp.nombre
            ORDER BY f30.fecha_venta DESC, f30.nombre;
        """)
        
        # Ejecutar query y obtener resultados
        logging.info("Ejecutando query para obtener datos de facturas y métricas...")
        with engine.connect() as conn:
            result = conn.execute(query_sql, {'fecha_limite': fecha_limite})
            rows = result.fetchall()
        
        if not rows:
            logging.warning("No se encontraron facturas en los últimos 30 días")
            return 0
        
        logging.info(f"Se encontraron {len(rows)} registros de facturas")
        
        # Truncar tabla antes de insertar nuevos datos (en transacción separada)
        logging.info(f"Truncando tabla {TABLE_NAME}...")
        with engine.begin() as conn:
            truncate_sql = text(f"TRUNCATE TABLE {TABLE_NAME};")
            conn.execute(truncate_sql)
        logging.info(f"Tabla {TABLE_NAME} truncada exitosamente")
        
        # Preparar datos para inserción
        dtype_mapping = {
            'nombre': sa_types.String(length=500),
            'precio': sa_types.NUMERIC(precision=12, scale=2),
            'cantidad': sa_types.INTEGER(),
            'metodo': sa_types.String(length=50),
            'vendedor': sa_types.String(length=100),
            'familia': sa_types.String(length=100),
            'precio_promedio_compra': sa_types.NUMERIC(precision=12, scale=2),
            'proveedor_moda': sa_types.String(length=300),
            'fecha_venta': sa_types.DATE()
        }
        
        # Convertir resultados a DataFrame
        logging.info("Convirtiendo resultados a DataFrame...")
        df = pd.DataFrame(rows, columns=[
            'nombre', 'precio', 'cantidad', 'metodo', 'vendedor',
            'familia', 'precio_promedio_compra', 'proveedor_moda', 'fecha_venta'
        ])
        
        # Convertir tipos de datos
        logging.info("Convirtiendo tipos de datos...")
        df['precio'] = pd.to_numeric(df['precio'], errors='coerce')
        df['cantidad'] = pd.to_numeric(df['cantidad'], errors='coerce').astype('Int64')
        df['precio_promedio_compra'] = pd.to_numeric(df['precio_promedio_compra'], errors='coerce')
        df['fecha_venta'] = pd.to_datetime(df['fecha_venta'], errors='coerce').dt.date
        
        # Insertar datos usando chunksize para evitar bloqueos
        logging.info(f"Insertando {len(df)} registros en {TABLE_NAME} (en lotes de 500)...")
        df.to_sql(
            TABLE_NAME,
            engine,
            if_exists='append',
            index=False,
            dtype=dtype_mapping,
            method='multi',
            chunksize=500
        )
        
        logging.info(f"✅ Reporte generado exitosamente: {len(df)} registros insertados")
        return len(df)
            
    except Exception as e:
        logging.error(f"Error generando reporte: {e}")
        raise ReporteError(f"Error en la generación del reporte: {e}")


def main():
    """Función principal del generador de reportes."""
    setup_logging(LOG_LEVEL)
    load_dotenv()
    
    logging.info("=== Iniciando generación de reporte de ventas de últimos 30 días ===")
    
    engine = None
    try:
        # Conectar a base de datos
        engine = get_database_engine()
        
        # Crear/reemplazar tabla
        create_report_table(engine)
        
        # Generar reporte
        registros = generate_report(engine)
        
        if registros > 0:
            logging.info(f"=== Proceso completado exitosamente: {registros} registros en {TABLE_NAME} ===")
        else:
            logging.info("=== Proceso completado: No hay registros para reportar ===")
        
        return 0
        
    except ReporteError as e:
        logging.error(f"Error del generador de reportes: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error inesperado: {e}")
        sys.exit(1)
    finally:
        # Cerrar todas las conexiones del engine
        if engine is not None:
            engine.dispose()
            logging.info("Conexiones a la base de datos cerradas")


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()

