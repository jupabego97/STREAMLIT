#!/usr/bin/env python3
"""
Clase Base para Extractores de Alegra
--------------------------------------

Este módulo define la clase base abstracta que todos los extractores
deben heredar para mantener consistencia en el código.

Uso:
    from base_extractor import BaseExtractor
    
    class MiExtractor(BaseExtractor):
        def extract(self):
            # Implementación específica
            pass
        
        def transform(self, data):
            # Implementación específica
            pass
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import aiohttp
import nest_asyncio
import pandas as pd
from sqlalchemy import text, types as sa_types
from sqlalchemy.engine import Engine

from config import settings, ConfigurationError
from utils import (
    get_database_engine,
    close_database_engine,
    setup_logging,
    fetch_with_retry,
)


class ExtractorError(Exception):
    """Excepción base para errores de extractores."""
    pass


class BaseExtractor(ABC):
    """
    Clase base abstracta para todos los extractores de Alegra.
    
    Proporciona funcionalidad común como:
    - Conexión a base de datos
    - Creación de tablas
    - Extracción concurrente
    - Manejo de errores estandarizado
    - Logging consistente
    
    Subclases deben implementar:
    - extract(): Lógica de extracción específica
    - transform(data): Transformación de datos
    - get_table_name(): Nombre de la tabla destino
    - get_dtype_mapping(): Mapeo de tipos de columnas
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Inicializa el extractor.
        
        Args:
            name: Nombre del extractor para logging.
        """
        self.name = name or self.__class__.__name__
        self.logger = setup_logging(self.name)
        self.engine: Optional[Engine] = None
        self._start_time: Optional[datetime] = None
    
    # -------------------------------------------------------------------------
    # Métodos abstractos (deben ser implementados por subclases)
    # -------------------------------------------------------------------------
    
    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """
        Extrae datos de la fuente (API).
        
        Returns:
            pd.DataFrame: Datos extraídos.
        """
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma los datos extraídos.
        
        Args:
            data: DataFrame con datos crudos.
        
        Returns:
            pd.DataFrame: Datos transformados.
        """
        pass
    
    @abstractmethod
    def get_table_name(self) -> str:
        """
        Retorna el nombre de la tabla destino.
        
        Returns:
            str: Nombre de la tabla.
        """
        pass
    
    @abstractmethod
    def get_dtype_mapping(self) -> Dict[str, Any]:
        """
        Retorna el mapeo de tipos de datos para SQLAlchemy.
        
        Returns:
            Dict[str, Any]: Mapeo columna -> tipo SQLAlchemy.
        """
        pass
    
    # -------------------------------------------------------------------------
    # Métodos de conexión
    # -------------------------------------------------------------------------
    
    def connect_database(self) -> bool:
        """
        Conecta a la base de datos PostgreSQL.
        
        Returns:
            bool: True si la conexión fue exitosa.
        """
        try:
            self.engine = get_database_engine()
            self.logger.info("Conexión a base de datos establecida")
            return True
        except (ConfigurationError, ConnectionError) as e:
            self.logger.error(f"Error conectando a la base de datos: {e}")
            return False
    
    def disconnect_database(self) -> None:
        """Cierra la conexión a la base de datos."""
        if self.engine:
            close_database_engine()
            self.engine = None
            self.logger.info("Conexión a base de datos cerrada")
    
    # -------------------------------------------------------------------------
    # Métodos de tabla
    # -------------------------------------------------------------------------
    
    def table_exists(self) -> bool:
        """
        Verifica si la tabla destino existe.
        
        Returns:
            bool: True si la tabla existe.
        """
        if not self.engine:
            return False
        
        table_name = self.get_table_name()
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = :table_name
            );
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), {'table_name': table_name})
                return result.scalar() or False
        except Exception as e:
            self.logger.error(f"Error verificando tabla: {e}")
            return False
    
    def create_table_if_not_exists(self, create_sql: str) -> bool:
        """
        Crea la tabla si no existe.
        
        Args:
            create_sql: SQL para crear la tabla.
        
        Returns:
            bool: True si la operación fue exitosa.
        """
        if not self.engine:
            self.logger.error("No hay conexión a la base de datos")
            return False
        
        table_name = self.get_table_name()
        
        try:
            if not self.table_exists():
                with self.engine.begin() as conn:
                    conn.execute(text(create_sql))
                self.logger.info(f"Tabla {table_name} creada exitosamente")
            else:
                self.logger.info(f"Tabla {table_name} ya existe")
            return True
        except Exception as e:
            self.logger.error(f"Error creando tabla {table_name}: {e}")
            return False
    
    def get_max_id(self, id_column: str = "id") -> Optional[int]:
        """
        Obtiene el ID máximo de la tabla.
        
        Args:
            id_column: Nombre de la columna ID.
        
        Returns:
            Optional[int]: ID máximo o None si la tabla está vacía.
        """
        if not self.engine or not self.table_exists():
            return None
        
        table_name = self.get_table_name()
        query = f"SELECT MAX({id_column}) as max_id FROM {table_name}"
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query)).scalar()
                if result:
                    self.logger.info(f"ID máximo en {table_name}: {result}")
                    return int(result)
                return None
        except Exception as e:
            self.logger.error(f"Error obteniendo ID máximo: {e}")
            return None
    
    def get_record_count(self) -> int:
        """
        Obtiene el número de registros en la tabla.
        
        Returns:
            int: Número de registros.
        """
        if not self.engine or not self.table_exists():
            return 0
        
        table_name = self.get_table_name()
        query = f"SELECT COUNT(*) FROM {table_name}"
        
        try:
            with self.engine.connect() as conn:
                return conn.execute(text(query)).scalar() or 0
        except Exception as e:
            self.logger.error(f"Error contando registros: {e}")
            return 0
    
    # -------------------------------------------------------------------------
    # Métodos de carga
    # -------------------------------------------------------------------------
    
    def load(self, df: pd.DataFrame, if_exists: str = "append") -> bool:
        """
        Carga datos en la base de datos.
        
        Args:
            df: DataFrame a cargar.
            if_exists: Comportamiento si la tabla existe ('append', 'replace').
        
        Returns:
            bool: True si la carga fue exitosa.
        """
        if df.empty:
            self.logger.info("No hay datos para cargar")
            return True
        
        if not self.engine:
            self.logger.error("No hay conexión a la base de datos")
            return False
        
        table_name = self.get_table_name()
        
        try:
            df.to_sql(
                table_name,
                self.engine,
                if_exists=if_exists,
                index=False,
                dtype=self.get_dtype_mapping(),
                method='multi',
                chunksize=500
            )
            self.logger.info(f"Cargados {len(df)} registros en {table_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error cargando datos en {table_name}: {e}")
            return False
    
    def truncate_table(self) -> bool:
        """
        Trunca (vacía) la tabla.
        
        Returns:
            bool: True si la operación fue exitosa.
        """
        if not self.engine:
            return False
        
        table_name = self.get_table_name()
        
        try:
            with self.engine.begin() as conn:
                conn.execute(text(f"TRUNCATE TABLE {table_name}"))
            self.logger.info(f"Tabla {table_name} truncada")
            return True
        except Exception as e:
            self.logger.error(f"Error truncando tabla: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Métodos de extracción concurrente
    # -------------------------------------------------------------------------
    
    async def fetch_page_async(
        self,
        session: aiohttp.ClientSession,
        url: str
    ) -> Any:
        """
        Descarga una página de manera asíncrona.
        
        Args:
            session: Sesión aiohttp.
            url: URL a descargar.
        
        Returns:
            Any: Datos de la respuesta.
        """
        return await fetch_with_retry(session, url)
    
    async def fetch_all_pages_async(
        self,
        urls: List[str],
        concurrency: Optional[int] = None
    ) -> List[Any]:
        """
        Descarga múltiples páginas de manera concurrente.
        
        Args:
            urls: Lista de URLs a descargar.
            concurrency: Número máximo de peticiones simultáneas.
        
        Returns:
            List[Any]: Lista de respuestas.
        """
        nest_asyncio.apply()
        
        max_concurrent = concurrency or settings.MAX_CONCURRENT_REQUESTS
        semaphore = asyncio.Semaphore(max_concurrent)
        headers = settings.get_api_headers()
        
        async def bounded_fetch(url: str) -> Any:
            async with semaphore:
                return await self.fetch_page_async(session, url)
        
        results = []
        async with aiohttp.ClientSession(headers=headers) as session:
            tasks = [bounded_fetch(url) for url in urls]
            results = await asyncio.gather(*tasks)
        
        return results
    
    def fetch_concurrent(self, urls: List[str]) -> List[Any]:
        """
        Wrapper síncrono para descarga concurrente.
        
        Args:
            urls: Lista de URLs a descargar.
        
        Returns:
            List[Any]: Lista de respuestas.
        """
        return asyncio.run(self.fetch_all_pages_async(urls))
    
    # -------------------------------------------------------------------------
    # Métodos de exportación
    # -------------------------------------------------------------------------
    
    def export_to_csv(self, filename: Optional[str] = None) -> bool:
        """
        Exporta los datos de la tabla a CSV.
        
        Args:
            filename: Nombre del archivo CSV.
        
        Returns:
            bool: True si la exportación fue exitosa.
        """
        if not self.engine or not self.table_exists():
            self.logger.warning("No hay datos para exportar")
            return False
        
        table_name = self.get_table_name()
        csv_filename = filename or f"{table_name}_backup.csv"
        
        try:
            query = f"SELECT * FROM {table_name} ORDER BY id"
            df = pd.read_sql(query, self.engine)
            df.to_csv(csv_filename, index=False)
            self.logger.info(f"Exportados {len(df)} registros a {csv_filename}")
            return True
        except Exception as e:
            self.logger.error(f"Error exportando a CSV: {e}")
            return False
    
    # -------------------------------------------------------------------------
    # Método principal de ejecución
    # -------------------------------------------------------------------------
    
    def run(self) -> bool:
        """
        Ejecuta el proceso completo de ETL.
        
        Returns:
            bool: True si el proceso fue exitoso.
        """
        self._start_time = datetime.now()
        self.logger.info(f"=== Iniciando {self.name} ===")
        
        try:
            # Validar configuración
            settings.validate()
            
            # Conectar a base de datos
            if not self.connect_database():
                raise ExtractorError("No se pudo conectar a la base de datos")
            
            # Extraer datos
            self.logger.info("Extrayendo datos...")
            raw_data = self.extract()
            
            if raw_data.empty:
                self.logger.info("No hay datos nuevos para procesar")
                return True
            
            # Transformar datos
            self.logger.info("Transformando datos...")
            transformed_data = self.transform(raw_data)
            
            if transformed_data.empty:
                self.logger.warning("No hay datos después de la transformación")
                return True
            
            # Cargar datos
            self.logger.info("Cargando datos...")
            if not self.load(transformed_data):
                raise ExtractorError("Error cargando datos")
            
            # Exportar a CSV si está configurado
            if settings.EXPORT_TO_CSV:
                self.export_to_csv()
            
            duration = datetime.now() - self._start_time
            self.logger.info(f"=== {self.name} completado en {duration} ===")
            return True
            
        except ConfigurationError as e:
            self.logger.error(f"Error de configuración: {e}")
            return False
        except ExtractorError as e:
            self.logger.error(f"Error del extractor: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error inesperado: {e}")
            return False
        finally:
            self.disconnect_database()
    
    # -------------------------------------------------------------------------
    # Utilidades
    # -------------------------------------------------------------------------
    
    @staticmethod
    def get_common_dtype_mapping() -> Dict[str, Any]:
        """
        Retorna mapeo de tipos comunes para SQLAlchemy.
        
        Returns:
            Dict[str, Any]: Mapeo de tipos comunes.
        """
        return {
            'id': sa_types.INTEGER(),
            'fecha': sa_types.DATE(),
            'nombre': sa_types.String(length=500),
            'precio': sa_types.NUMERIC(precision=12, scale=2),
            'cantidad': sa_types.INTEGER(),
            'total': sa_types.NUMERIC(precision=12, scale=2),
        }


if __name__ == "__main__":
    # Este archivo no se ejecuta directamente
    print("Este módulo define la clase base para extractores.")
    print("Úsalo heredando de BaseExtractor en tus extractores específicos.")

