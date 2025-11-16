#!/usr/bin/env python3
"""
Generador de Tabla para_pedidos
--------------------------------

▶ **Funcionalidades principales:**
   1. Genera tabla para_pedidos con información consolidada para gestión de inventario
   2. Incluye datos de items, facturas_proveedor y facturas
   3. Calcula ventas por períodos (7, 15, 30, 60, 90 días)
   4. Calcula métricas de compra (precio promedio, moda proveedor, última compra)
   5. Calcula cantidad recomendada a comprar basada en ventas de últimos 12 meses
   6. Calcula margen y utilidad por producto
   7. Reemplaza tabla completa cada ejecución (TRUNCATE + INSERT)

▶ **Datos consolidados:**
   - De items: nombre, familia, cantidad_disponible
   - De facturas_proveedor: precio_promedio_compra, moda_proveedor, fecha_ultima_compra, 
     cantidad_ultima_compra, precio_ultimo_compra
   - De facturas: ventas por períodos y precio promedio de últimas 5 ventas
   - Cálculos: cantidad_a_comprar, margen, utilidad

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
TABLE_NAME = "para_pedidos"
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


class PedidosError(Exception):
    """Excepción personalizada para errores del generador de pedidos."""
    pass


def get_database_engine():
    """Crea el engine de SQLAlchemy para PostgreSQL."""
    if not SQLALCHEMY_AVAILABLE:
        raise PedidosError("SQLAlchemy no está disponible. Instala: pip install sqlalchemy")
    
    db_url = os.getenv(DB_URL_ENV)
    if not db_url:
        raise PedidosError(f"Variable {DB_URL_ENV} no encontrada. Configura la URL de la base de datos.")
    
    try:
        engine = create_engine(db_url)
        # Probar conexión
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logging.info("Conexión a PostgreSQL establecida")
        return engine
    except Exception as e:
        raise PedidosError(f"No se pudo conectar a PostgreSQL: {e}")


def create_table_para_pedidos(engine):
    """Crea la tabla para_pedidos si no existe."""
    if engine is None:
        raise PedidosError("No hay conexión a la base de datos")
    
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
                        familia VARCHAR(100),
                        cantidad_disponible DECIMAL(10,2),
                        
                        -- De facturas_proveedor
                        precio_promedio_compra DECIMAL(12,2),
                        moda_proveedor VARCHAR(300),
                        fecha_ultima_compra DATE,
                        cantidad_ultima_compra DECIMAL(10,2),
                        precio_ultimo_compra DECIMAL(12,2),
                        
                        -- De facturas (ventas por período)
                        ventas_90_dias INTEGER,
                        ventas_60_dias INTEGER,
                        ventas_30_dias INTEGER,
                        ventas_15_dias INTEGER,
                        ventas_7_dias INTEGER,
                        
                        -- Cálculos
                        promedio_ventas_12_meses DECIMAL(10,2),
                        cantidad_a_comprar DECIMAL(10,2),
                        precio_promedio_venta DECIMAL(12,2),
                        margen DECIMAL(12,2),
                        utilidad DECIMAL(5,2),
                        
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE INDEX idx_{TABLE_NAME}_nombre ON {TABLE_NAME}(nombre);
                    CREATE INDEX idx_{TABLE_NAME}_familia ON {TABLE_NAME}(familia);
                """)
                conn.execute(create_table_sql)
                logging.info(f"Tabla {TABLE_NAME} creada exitosamente")
            else:
                logging.info(f"Tabla {TABLE_NAME} ya existe")
            return True
            
    except Exception as e:
        logging.error(f"Error creando/verificando tabla {TABLE_NAME}: {e}")
        raise PedidosError(f"No se pudo crear/verificar la tabla: {e}")


def generate_para_pedidos(engine):
    """Genera la tabla para_pedidos con todos los cálculos requeridos."""
    if engine is None:
        raise PedidosError("No hay conexión a la base de datos")
    
    logging.info("Generando tabla para_pedidos con cálculos consolidados...")
    
    try:
        # Query principal usando CTEs para todos los cálculos
        query_sql = text("""
            WITH items_base AS (
                -- Datos base de items
                SELECT 
                    nombre,
                    familia,
                    cantidad_disponible
                FROM items
            ),
            compras_ultimas_3_meses AS (
                -- Últimas 3 compras de los últimos 3 meses por producto
                SELECT 
                    fp.nombre,
                    fp.precio,
                    fp.proveedor,
                    fp.fecha,
                    fp.cantidad,
                    ROW_NUMBER() OVER (PARTITION BY fp.nombre ORDER BY fp.fecha DESC) as rn
                FROM facturas_proveedor fp
                WHERE fp.fecha >= CURRENT_DATE - INTERVAL '3 months'
            ),
            todas_compras_ordenadas AS (
                -- Todas las compras ordenadas por fecha (para usar última si no hay compras en últimos 3 meses)
                SELECT 
                    fp.nombre,
                    fp.precio,
                    fp.proveedor,
                    fp.fecha,
                    fp.cantidad,
                    ROW_NUMBER() OVER (PARTITION BY fp.nombre ORDER BY fp.fecha DESC) as rn
                FROM facturas_proveedor fp
            ),
            productos_con_compras_3_meses AS (
                -- Productos que tienen al menos una compra en los últimos 3 meses
                SELECT DISTINCT nombre
                FROM compras_ultimas_3_meses
            ),
            compras_relevantes AS (
                -- Usar últimas 3 compras de últimos 3 meses si existen (o todas si hay menos de 3)
                SELECT 
                    c3m.nombre,
                    c3m.precio,
                    c3m.proveedor,
                    c3m.fecha,
                    c3m.cantidad
                FROM compras_ultimas_3_meses c3m
                WHERE c3m.rn <= 3
                UNION ALL
                -- Para productos sin compras en últimos 3 meses, usar la última compra disponible
                SELECT 
                    tco.nombre,
                    tco.precio,
                    tco.proveedor,
                    tco.fecha,
                    tco.cantidad
                FROM todas_compras_ordenadas tco
                WHERE tco.rn = 1
                  AND tco.nombre NOT IN (SELECT nombre FROM productos_con_compras_3_meses)
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
                    proveedor AS moda_proveedor
                FROM proveedor_conteo
                ORDER BY nombre, frecuencia DESC, proveedor
            ),
            precio_promedio_compras AS (
                -- Calcular precio promedio de compras relevantes (últimas 3 o última disponible)
                SELECT 
                    nombre,
                    AVG(precio) AS precio_promedio_compra
                FROM compras_relevantes
                GROUP BY nombre
            ),
            ultima_compra AS (
                -- Obtener la última compra por producto (sin restricción de tiempo)
                SELECT DISTINCT ON (nombre)
                    nombre,
                    fecha AS fecha_ultima_compra,
                    cantidad AS cantidad_ultima_compra,
                    precio AS precio_ultimo_compra
                FROM facturas_proveedor
                ORDER BY nombre, fecha DESC
            ),
            metricas_compras AS (
                -- Unir todas las métricas de compras
                SELECT 
                    COALESCE(ppc.nombre, uc.nombre) AS nombre,
                    ppc.precio_promedio_compra,
                    pr.moda_proveedor,
                    uc.fecha_ultima_compra,
                    uc.cantidad_ultima_compra,
                    uc.precio_ultimo_compra
                FROM precio_promedio_compras ppc
                FULL OUTER JOIN ultima_compra uc ON ppc.nombre = uc.nombre
                LEFT JOIN proveedor_ranking pr ON COALESCE(ppc.nombre, uc.nombre) = pr.nombre
            ),
            ventas_por_periodo AS (
                -- Calcular ventas por períodos desde facturas
                SELECT 
                    f.nombre,
                    SUM(CASE WHEN f.fecha >= CURRENT_DATE - INTERVAL '90 days' THEN f.cantidad ELSE 0 END)::INTEGER AS ventas_90_dias,
                    SUM(CASE WHEN f.fecha >= CURRENT_DATE - INTERVAL '60 days' THEN f.cantidad ELSE 0 END)::INTEGER AS ventas_60_dias,
                    SUM(CASE WHEN f.fecha >= CURRENT_DATE - INTERVAL '30 days' THEN f.cantidad ELSE 0 END)::INTEGER AS ventas_30_dias,
                    SUM(CASE WHEN f.fecha >= CURRENT_DATE - INTERVAL '15 days' THEN f.cantidad ELSE 0 END)::INTEGER AS ventas_15_dias,
                    SUM(CASE WHEN f.fecha >= CURRENT_DATE - INTERVAL '7 days' THEN f.cantidad ELSE 0 END)::INTEGER AS ventas_7_dias
                FROM facturas f
                GROUP BY f.nombre
            ),
            promedios_diarios_ventas AS (
                -- Calcular promedios diarios de ventas para algoritmo de retail
                SELECT 
                    f.nombre,
                    -- Promedio diario últimos 30 días
                    CASE 
                        WHEN COUNT(CASE WHEN f.fecha >= CURRENT_DATE - INTERVAL '30 days' THEN 1 END) > 0 
                        THEN SUM(CASE WHEN f.fecha >= CURRENT_DATE - INTERVAL '30 days' THEN f.cantidad ELSE 0 END)::DECIMAL / 30.0
                        ELSE 0 
                    END AS promedio_diario_30_dias,
                    -- Promedio diario últimos 60 días
                    CASE 
                        WHEN COUNT(CASE WHEN f.fecha >= CURRENT_DATE - INTERVAL '60 days' THEN 1 END) > 0 
                        THEN SUM(CASE WHEN f.fecha >= CURRENT_DATE - INTERVAL '60 days' THEN f.cantidad ELSE 0 END)::DECIMAL / 60.0
                        ELSE 0 
                    END AS promedio_diario_60_dias,
                    -- Promedio diario últimos 90 días
                    CASE 
                        WHEN COUNT(CASE WHEN f.fecha >= CURRENT_DATE - INTERVAL '90 days' THEN 1 END) > 0 
                        THEN SUM(CASE WHEN f.fecha >= CURRENT_DATE - INTERVAL '90 days' THEN f.cantidad ELSE 0 END)::DECIMAL / 90.0
                        ELSE 0 
                    END AS promedio_diario_90_dias
                FROM facturas f
                GROUP BY f.nombre
            ),
            ventas_12_meses AS (
                -- Calcular promedio diario de ventas en últimos 12 meses
                SELECT 
                    f.nombre,
                    COALESCE(SUM(f.cantidad)::DECIMAL / 365.0, 0) AS promedio_ventas_12_meses
                FROM facturas f
                WHERE f.fecha >= CURRENT_DATE - INTERVAL '12 months'
                GROUP BY f.nombre
            ),
            ultimas_5_ventas AS (
                -- Obtener últimas 5 ventas por producto para calcular precio promedio
                SELECT 
                    f.nombre,
                    f.precio,
                    f.fecha,
                    ROW_NUMBER() OVER (PARTITION BY f.nombre ORDER BY f.fecha DESC) as rn
                FROM facturas f
            ),
            precio_promedio_venta AS (
                -- Calcular precio promedio de últimas 5 ventas
                SELECT 
                    nombre,
                    AVG(precio) AS precio_promedio_venta
                FROM ultimas_5_ventas
                WHERE rn <= 5
                GROUP BY nombre
            ),
            datos_consolidados AS (
                -- Unir todos los datos
                SELECT 
                    i.nombre,
                    i.familia,
                    i.cantidad_disponible,
                    mc.precio_promedio_compra,
                    mc.moda_proveedor,
                    mc.fecha_ultima_compra,
                    mc.cantidad_ultima_compra,
                    mc.precio_ultimo_compra,
                    COALESCE(vpp.ventas_90_dias, 0) AS ventas_90_dias,
                    COALESCE(vpp.ventas_60_dias, 0) AS ventas_60_dias,
                    COALESCE(vpp.ventas_30_dias, 0) AS ventas_30_dias,
                    COALESCE(vpp.ventas_15_dias, 0) AS ventas_15_dias,
                    COALESCE(vpp.ventas_7_dias, 0) AS ventas_7_dias,
                    COALESCE(v12.promedio_ventas_12_meses, 0) AS promedio_ventas_12_meses,
                    ppv.precio_promedio_venta,
                    pdv.promedio_diario_30_dias,
                    pdv.promedio_diario_60_dias,
                    pdv.promedio_diario_90_dias
                FROM items_base i
                LEFT JOIN metricas_compras mc ON i.nombre = mc.nombre
                LEFT JOIN ventas_por_periodo vpp ON i.nombre = vpp.nombre
                LEFT JOIN ventas_12_meses v12 ON i.nombre = v12.nombre
                LEFT JOIN precio_promedio_venta ppv ON i.nombre = ppv.nombre
                LEFT JOIN promedios_diarios_ventas pdv ON i.nombre = pdv.nombre
            )
            -- Calcular campos finales con algoritmo mejorado de retail
            SELECT 
                nombre,
                familia,
                cantidad_disponible,
                precio_promedio_compra,
                moda_proveedor,
                fecha_ultima_compra,
                cantidad_ultima_compra,
                precio_ultimo_compra,
                ventas_90_dias,
                ventas_60_dias,
                ventas_30_dias,
                ventas_15_dias,
                ventas_7_dias,
                promedio_ventas_12_meses,
                -- Algoritmo de promedio móvil ponderado con tendencia para cantidad_a_comprar
                GREATEST(
                    CASE 
                        -- Si hay datos de ventas, usar algoritmo de retail
                        WHEN promedio_diario_30_dias > 0 OR promedio_diario_60_dias > 0 OR promedio_diario_90_dias > 0 THEN
                            (
                                -- Promedio ponderado: 50% últimos 30 días, 30% últimos 60 días, 20% últimos 90 días
                                (COALESCE(promedio_diario_30_dias, 0) * 0.5) +
                                (COALESCE(promedio_diario_60_dias, 0) * 0.3) +
                                (COALESCE(promedio_diario_90_dias, 0) * 0.2)
                            ) * 30.0 * 
                            -- Factor de tendencia: calcular crecimiento/decrecimiento
                            GREATEST(0.7, LEAST(1.3, 
                                CASE 
                                    WHEN promedio_diario_60_dias > 0 THEN
                                        1.0 + ((promedio_diario_30_dias - promedio_diario_60_dias) / NULLIF(promedio_diario_60_dias, 0) * 0.5)
                                    ELSE 1.0
                                END
                            ))
                        -- Si no hay datos recientes, usar promedio de 12 meses
                        WHEN promedio_ventas_12_meses > 0 THEN
                            promedio_ventas_12_meses * 30.0
                        ELSE 0
                    END - COALESCE(cantidad_disponible, 0),
                    0
                ) AS cantidad_a_comprar,
                precio_promedio_venta,
                precio_promedio_venta - precio_ultimo_compra AS margen,
                CASE 
                    WHEN precio_ultimo_compra > 0 THEN 
                        ((precio_promedio_venta - precio_ultimo_compra) / precio_ultimo_compra * 100)
                    ELSE NULL
                END AS utilidad
            FROM datos_consolidados
            ORDER BY nombre;
        """)
        
        # Ejecutar query y obtener resultados
        logging.info("Ejecutando query compleja para obtener datos consolidados...")
        try:
            with engine.connect() as conn:
                result = conn.execute(query_sql)
                rows = result.fetchall()
                logging.info(f"Query ejecutada exitosamente. Filas obtenidas: {len(rows)}")
        except Exception as e:
            logging.error(f"Error ejecutando query SQL: {e}")
            logging.error(f"Query que falló: {query_sql}")
            raise PedidosError(f"Error en la ejecución de la query: {e}")
        
        if not rows:
            logging.warning("No se encontraron items para procesar. Verificar que existan datos en la tabla 'items'.")
            # Verificar si hay items en la base de datos
            with engine.connect() as conn:
                count_items = conn.execute(text("SELECT COUNT(*) FROM items")).scalar()
                logging.info(f"Total de items en la base de datos: {count_items}")
            return 0
        
        logging.info(f"Se encontraron {len(rows)} productos para procesar")
        logging.info(f"Primera fila de ejemplo: {rows[0] if rows else 'N/A'}")
        
        # Truncar tabla antes de insertar nuevos datos
        logging.info(f"Truncando tabla {TABLE_NAME}...")
        with engine.begin() as conn:
            truncate_sql = text(f"TRUNCATE TABLE {TABLE_NAME};")
            conn.execute(truncate_sql)
        logging.info(f"Tabla {TABLE_NAME} truncada exitosamente")
        
        # Preparar datos para inserción
        dtype_mapping = {
            'nombre': sa_types.String(length=500),
            'familia': sa_types.String(length=100),
            'cantidad_disponible': sa_types.NUMERIC(precision=10, scale=2),
            'precio_promedio_compra': sa_types.NUMERIC(precision=12, scale=2),
            'moda_proveedor': sa_types.String(length=300),
            'fecha_ultima_compra': sa_types.DATE(),
            'cantidad_ultima_compra': sa_types.NUMERIC(precision=10, scale=2),
            'precio_ultimo_compra': sa_types.NUMERIC(precision=12, scale=2),
            'ventas_90_dias': sa_types.INTEGER(),
            'ventas_60_dias': sa_types.INTEGER(),
            'ventas_30_dias': sa_types.INTEGER(),
            'ventas_15_dias': sa_types.INTEGER(),
            'ventas_7_dias': sa_types.INTEGER(),
            'promedio_ventas_12_meses': sa_types.NUMERIC(precision=10, scale=2),
            'cantidad_a_comprar': sa_types.NUMERIC(precision=10, scale=2),
            'precio_promedio_venta': sa_types.NUMERIC(precision=12, scale=2),
            'margen': sa_types.NUMERIC(precision=12, scale=2),
            'utilidad': sa_types.NUMERIC(precision=5, scale=2)
        }
        
        # Convertir resultados a DataFrame
        logging.info("Convirtiendo resultados a DataFrame...")
        try:
            df = pd.DataFrame(rows, columns=[
                'nombre', 'familia', 'cantidad_disponible',
                'precio_promedio_compra', 'moda_proveedor', 'fecha_ultima_compra',
                'cantidad_ultima_compra', 'precio_ultimo_compra',
                'ventas_90_dias', 'ventas_60_dias', 'ventas_30_dias',
                'ventas_15_dias', 'ventas_7_dias',
                'promedio_ventas_12_meses', 'cantidad_a_comprar',
                'precio_promedio_venta', 'margen', 'utilidad'
            ])
            logging.info(f"DataFrame creado: {len(df)} filas, {len(df.columns)} columnas")
            logging.info(f"Columnas del DataFrame: {list(df.columns)}")
        except Exception as e:
            logging.error(f"Error creando DataFrame: {e}")
            logging.error(f"Número de columnas esperadas: 18, número de columnas en rows[0]: {len(rows[0]) if rows else 0}")
            if rows:
                logging.error(f"Primera fila tiene {len(rows[0])} elementos: {rows[0]}")
            raise PedidosError(f"Error creando DataFrame: {e}")
        
        # Convertir tipos de datos
        logging.info("Convirtiendo tipos de datos...")
        numeric_cols = ['cantidad_disponible', 'precio_promedio_compra', 'cantidad_ultima_compra',
                       'precio_ultimo_compra', 'promedio_ventas_12_meses', 'cantidad_a_comprar',
                       'precio_promedio_venta', 'margen', 'utilidad']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        integer_cols = ['ventas_90_dias', 'ventas_60_dias', 'ventas_30_dias', 'ventas_15_dias', 'ventas_7_dias']
        for col in integer_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        
        if 'fecha_ultima_compra' in df.columns:
            df['fecha_ultima_compra'] = pd.to_datetime(df['fecha_ultima_compra'], errors='coerce').dt.date
        
        # Asegurar que cantidad_a_comprar no sea negativa
        df['cantidad_a_comprar'] = df['cantidad_a_comprar'].clip(lower=0)
        
        # Validar y limpiar datos antes de insertar
        logging.info("Validando y limpiando datos antes de insertar...")
        
        # Asegurar que 'nombre' no sea NULL (es NOT NULL en la tabla)
        if df['nombre'].isna().any():
            logging.warning(f"Se encontraron {df['nombre'].isna().sum()} registros con nombre NULL. Eliminándolos...")
            df = df.dropna(subset=['nombre'])
        
        # Truncar strings que excedan los límites de VARCHAR
        if 'nombre' in df.columns:
            df['nombre'] = df['nombre'].astype(str).str[:500]
        if 'familia' in df.columns:
            df['familia'] = df['familia'].astype(str).str[:100].replace('nan', None)
        if 'moda_proveedor' in df.columns:
            df['moda_proveedor'] = df['moda_proveedor'].astype(str).str[:300].replace('nan', None)
        
        # Reemplazar 'nan' strings con None para campos opcionales
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].replace('nan', None).replace('None', None)
        
        # Asegurar que los valores numéricos estén en rangos válidos
        # Reemplazar infinitos y valores muy grandes con None
        numeric_cols = ['cantidad_disponible', 'precio_promedio_compra', 'cantidad_ultima_compra',
                       'precio_ultimo_compra', 'promedio_ventas_12_meses', 'cantidad_a_comprar',
                       'precio_promedio_venta', 'margen', 'utilidad']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].replace([float('inf'), float('-inf')], None)
                # Limitar valores muy grandes
                if col in ['precio_promedio_compra', 'precio_ultimo_compra', 'precio_promedio_venta', 'margen']:
                    df[col] = df[col].clip(lower=-999999999.99, upper=999999999.99)
                elif col == 'utilidad':
                    df[col] = df[col].clip(lower=-999.99, upper=999.99)
        
        # Validar que los enteros no tengan decimales
        integer_cols = ['ventas_90_dias', 'ventas_60_dias', 'ventas_30_dias', 'ventas_15_dias', 'ventas_7_dias']
        for col in integer_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype('Int64')
        
        logging.info(f"Datos validados. Filas restantes: {len(df)}")
        
        if len(df) == 0:
            logging.warning("No hay datos válidos para insertar después de la validación")
            return 0
        
        # Insertar datos
        logging.info(f"Insertando {len(df)} registros en {TABLE_NAME}...")
        logging.info(f"Primeras 3 filas del DataFrame:\n{df.head(3)}")
        logging.info(f"Shape del DataFrame: {df.shape}")
        logging.info(f"Tipos de datos del DataFrame:\n{df.dtypes}")
        
        try:
            # Intentar inserción con method='multi' primero
            try:
                df.to_sql(
                    TABLE_NAME,
                    engine,
                    if_exists='append',
                    index=False,
                    dtype=dtype_mapping,
                    method='multi',
                    chunksize=500
                )
                logging.info("Inserción exitosa usando method='multi'")
            except Exception as e1:
                logging.warning(f"Error con method='multi': {e1}. Intentando sin method...")
                logging.warning(f"Detalles del error: {type(e1).__name__}: {str(e1)}")
                import traceback
                logging.warning(f"Traceback parcial:\n{''.join(traceback.format_exc().splitlines()[-5:])}")
                # Si falla, intentar sin method='multi' y con chunksize más pequeño
                try:
                    df.to_sql(
                        TABLE_NAME,
                        engine,
                        if_exists='append',
                        index=False,
                        dtype=dtype_mapping,
                        chunksize=100  # Chunksize más pequeño
                    )
                    logging.info("Inserción exitosa sin method='multi'")
                except Exception as e2:
                    logging.error(f"Error también sin method='multi': {e2}")
                    logging.error(f"Tipo de error: {type(e2).__name__}")
                    # Mostrar información detallada del error
                    import traceback
                    logging.error(f"Traceback completo:\n{traceback.format_exc()}")
                    # Intentar identificar qué fila causa el problema
                    logging.info("Intentando identificar la fila problemática...")
                    logging.info(f"Primera fila del DataFrame:\n{df.iloc[0].to_dict()}")
                    raise e2
            
            logging.info(f"✅ Tabla {TABLE_NAME} generada exitosamente: {len(df)} registros insertados")
            
            # Verificar que los datos se insertaron correctamente
            with engine.connect() as conn:
                count_inserted = conn.execute(text(f"SELECT COUNT(*) FROM {TABLE_NAME}")).scalar()
                logging.info(f"Verificación: {count_inserted} registros en la tabla {TABLE_NAME}")
                
                if count_inserted == 0:
                    logging.error("⚠️ ADVERTENCIA: La inserción reportó éxito pero la tabla está vacía!")
                    # Intentar ver si hay algún problema con los datos
                    sample_query = text(f"SELECT * FROM {TABLE_NAME} LIMIT 5")
                    sample_rows = conn.execute(sample_query).fetchall()
                    logging.info(f"Filas de muestra en la tabla: {sample_rows}")
            
            return len(df)
        except Exception as e:
            logging.error(f"Error insertando datos en {TABLE_NAME}: {e}")
            logging.error(f"Tipo de error: {type(e).__name__}")
            import traceback
            logging.error(f"Traceback completo:\n{traceback.format_exc()}")
            raise PedidosError(f"Error insertando datos: {e}")
            
    except Exception as e:
        logging.error(f"Error generando tabla {TABLE_NAME}: {e}")
        raise PedidosError(f"Error en la generación de la tabla: {e}")


def main():
    """Función principal del generador de tabla para_pedidos."""
    setup_logging(LOG_LEVEL)
    load_dotenv()
    
    logging.info("=== Iniciando generación de tabla para_pedidos ===")
    
    engine = None
    try:
        # Conectar a base de datos
        engine = get_database_engine()
        
        # Crear/verificar tabla
        create_table_para_pedidos(engine)
        
        # Generar tabla
        registros = generate_para_pedidos(engine)
        
        if registros > 0:
            logging.info(f"=== Proceso completado exitosamente: {registros} registros en {TABLE_NAME} ===")
        else:
            logging.info("=== Proceso completado: No hay registros para procesar ===")
        
        return 0
        
    except PedidosError as e:
        logging.error(f"Error del generador de pedidos: {e}")
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

