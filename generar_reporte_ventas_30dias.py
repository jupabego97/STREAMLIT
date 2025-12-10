#!/usr/bin/env python3
"""
Generador de Reportes de Ventas de Últimos 30 Días
---------------------------------------------------

Genera una tabla consolidada con datos de ventas de los últimos 30 días,
incluyendo información de familia, precio promedio de compra y proveedor.

Funcionalidades:
- Genera tabla de reportes de facturas de ventas
- Incluye información de familia desde items
- Calcula precio promedio de compra (últimas 3 compras)
- Calcula proveedor moda (últimas 3 compras)
- Reemplaza tabla completa cada ejecución (TRUNCATE + INSERT)
- Crea índices optimizados para consultas frecuentes

Uso:
    python generar_reporte_ventas_30dias.py
"""
from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from sqlalchemy import text, types as sa_types

from config import settings, ConfigurationError
from utils import get_database_engine, close_database_engine, setup_logging

# Configurar logging
logger = setup_logging("ReporteVentas")


class ReporteError(Exception):
    """Excepción personalizada para errores del generador de reportes."""
    pass


class ReporteVentasGenerator:
    """
    Generador de reportes de ventas de los últimos 30 días.
    
    Consolida datos de facturas, items y facturas_proveedor
    en una tabla optimizada para análisis.
    """
    
    def __init__(self):
        self.engine = None
        self.table_name = settings.TABLE_REPORTES
    
    def connect(self) -> bool:
        """Conecta a la base de datos."""
        try:
            self.engine = get_database_engine()
            logger.info("Conexión a PostgreSQL establecida")
            return True
        except (ConfigurationError, ConnectionError) as e:
            logger.error(f"Error conectando a la base de datos: {e}")
            return False
    
    def disconnect(self) -> None:
        """Cierra la conexión a la base de datos."""
        if self.engine:
            close_database_engine()
            self.engine = None
            logger.info("Conexiones cerradas")
    
    def create_table_if_not_exists(self) -> bool:
        """Crea la tabla de reportes si no existe, con índices optimizados."""
        if not self.engine:
            raise ReporteError("No hay conexión a la base de datos")
        
        try:
            with self.engine.begin() as conn:
                # Verificar si la tabla existe
                check_sql = text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = :table_name
                    );
                """)
                table_exists = conn.execute(check_sql, {'table_name': self.table_name}).scalar()
                
                if not table_exists:
                    # Crear tabla con índices optimizados
                    create_sql = text(f"""
                        CREATE TABLE {self.table_name} (
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
                        
                        -- Índices para consultas frecuentes
                        CREATE INDEX idx_{self.table_name}_nombre 
                            ON {self.table_name}(nombre);
                        CREATE INDEX idx_{self.table_name}_fecha_venta 
                            ON {self.table_name}(fecha_venta);
                        CREATE INDEX idx_{self.table_name}_fecha_vendedor 
                            ON {self.table_name}(fecha_venta, vendedor);
                        CREATE INDEX idx_{self.table_name}_familia 
                            ON {self.table_name}(familia);
                        CREATE INDEX idx_{self.table_name}_vendedor 
                            ON {self.table_name}(vendedor);
                        CREATE INDEX idx_{self.table_name}_metodo 
                            ON {self.table_name}(metodo);
                    """)
                    conn.execute(create_sql)
                    logger.info(f"Tabla {self.table_name} creada con índices optimizados")
                else:
                    # Verificar y crear índices faltantes
                    self._ensure_indexes(conn)
                    logger.info(f"Tabla {self.table_name} ya existe - índices verificados")
                
                return True
                
        except Exception as e:
            logger.error(f"Error creando/verificando tabla: {e}")
            raise ReporteError(f"No se pudo crear/verificar la tabla: {e}")
    
    def _ensure_indexes(self, conn) -> None:
        """Asegura que existan los índices optimizados."""
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_nombre ON {self.table_name}(nombre)",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_fecha_venta ON {self.table_name}(fecha_venta)",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_fecha_vendedor ON {self.table_name}(fecha_venta, vendedor)",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_familia ON {self.table_name}(familia)",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_vendedor ON {self.table_name}(vendedor)",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_metodo ON {self.table_name}(metodo)",
        ]
        
        for idx_sql in indexes:
            try:
                conn.execute(text(idx_sql))
            except Exception as e:
                logger.warning(f"Error creando índice: {e}")
    
    def generate_report(self) -> int:
        """
        Genera el reporte de ventas de los últimos 30 días.
        
        Returns:
            int: Número de registros insertados.
        """
        if not self.engine:
            raise ReporteError("No hay conexión a la base de datos")
        
        fecha_limite = date.today() - timedelta(days=30)
        logger.info(f"Generando reporte desde {fecha_limite} hasta {date.today()}")
        
        try:
            # Query principal con CTEs para calcular métricas
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
            
            # Ejecutar query
            logger.info("Ejecutando query para obtener datos...")
            with self.engine.connect() as conn:
                result = conn.execute(query_sql, {'fecha_limite': fecha_limite})
                rows = result.fetchall()
            
            if not rows:
                logger.warning("No se encontraron facturas en los últimos 30 días")
                return 0
            
            logger.info(f"Se encontraron {len(rows)} registros")
            
            # Truncar tabla
            logger.info(f"Truncando tabla {self.table_name}...")
            with self.engine.begin() as conn:
                conn.execute(text(f"TRUNCATE TABLE {self.table_name};"))
            
            # Convertir a DataFrame
            df = pd.DataFrame(rows, columns=[
                'nombre', 'precio', 'cantidad', 'metodo', 'vendedor',
                'familia', 'precio_promedio_compra', 'proveedor_moda', 'fecha_venta'
            ])
            
            # Convertir tipos
            df['precio'] = pd.to_numeric(df['precio'], errors='coerce')
            df['cantidad'] = pd.to_numeric(df['cantidad'], errors='coerce').astype('Int64')
            df['precio_promedio_compra'] = pd.to_numeric(df['precio_promedio_compra'], errors='coerce')
            df['fecha_venta'] = pd.to_datetime(df['fecha_venta'], errors='coerce').dt.date
            
            # Mapeo de tipos para SQLAlchemy
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
            
            # Insertar datos
            logger.info(f"Insertando {len(df)} registros (en lotes de 500)...")
            df.to_sql(
                self.table_name,
                self.engine,
                if_exists='append',
                index=False,
                dtype=dtype_mapping,
                method='multi',
                chunksize=500
            )
            
            logger.info(f"✅ Reporte generado: {len(df)} registros insertados")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error generando reporte: {e}")
            raise ReporteError(f"Error en la generación del reporte: {e}")
    
    def run(self) -> int:
        """
        Ejecuta el proceso completo de generación de reportes.
        
        Returns:
            int: Número de registros generados.
        """
        logger.info("=== Iniciando generación de reporte de ventas ===")
        
        try:
            # Validar configuración (solo necesita DB, no API)
            settings.validate_for_dashboard()
            
            # Conectar
            if not self.connect():
                raise ReporteError("No se pudo conectar a la base de datos")
            
            # Crear/verificar tabla
            self.create_table_if_not_exists()
            
            # Generar reporte
            registros = self.generate_report()
            
            if registros > 0:
                logger.info(f"=== Proceso completado: {registros} registros ===")
            else:
                logger.info("=== Proceso completado: Sin registros ===")
            
            return registros
            
        except ReporteError as e:
            logger.error(f"Error del generador: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error inesperado: {e}")
            return 0
        finally:
            self.disconnect()


def main():
    """Función principal."""
    generator = ReporteVentasGenerator()
    
    try:
        registros = generator.run()
        sys.exit(0 if registros >= 0 else 1)
    except KeyboardInterrupt:
        logger.info("Proceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
