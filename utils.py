#!/usr/bin/env python3
"""
Utilidades Compartidas para la Aplicación STREAMLIT
----------------------------------------------------

Este módulo contiene funciones y clases reutilizables para:
- Conexión a base de datos
- Cliente HTTP para API
- Funciones asíncronas con reintentos
- Configuración de logging

Uso:
    from utils import get_database_engine, create_api_session, setup_logging
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, TypeVar

import aiohttp
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

from config import settings, ConfigurationError

# Type variable para funciones genéricas
T = TypeVar('T')


# =============================================================================
# Logging
# =============================================================================

def setup_logging(
    name: str = "app",
    level: Optional[int] = None,
    log_to_file: bool = False,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Configura el sistema de logging con formato consistente.
    
    Args:
        name: Nombre del logger.
        level: Nivel de logging (usa settings.LOG_LEVEL si no se especifica).
        log_to_file: Si True, también escribe a archivo con rotación.
        log_dir: Directorio para archivos de log.
    
    Returns:
        logging.Logger: Logger configurado.
    """
    logger = logging.getLogger(name)
    
    # Evitar configurar múltiples veces
    if logger.handlers:
        return logger
    
    log_level = level or settings.LOG_LEVEL
    logger.setLevel(log_level)
    
    # Formato
    formatter = logging.Formatter(
        settings.LOG_FORMAT,
        datefmt=settings.LOG_DATE_FORMAT
    )
    
    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    
    # Handler para archivo (opcional, con rotación)
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    
    return logger


# Logger por defecto del módulo
logger = setup_logging("utils")


# =============================================================================
# Base de Datos
# =============================================================================

_engine_cache: Optional[Engine] = None


def get_database_engine(force_new: bool = False) -> Engine:
    """
    Crea o retorna el engine de SQLAlchemy para PostgreSQL.
    
    Usa un cache para reutilizar la conexión y un pool de conexiones
    para mejor rendimiento.
    
    Args:
        force_new: Si True, crea un nuevo engine ignorando el cache.
    
    Returns:
        Engine: Engine de SQLAlchemy configurado.
    
    Raises:
        ConfigurationError: Si DATABASE_URL no está configurada.
        ConnectionError: Si no se puede conectar a la base de datos.
    """
    global _engine_cache
    
    if _engine_cache is not None and not force_new:
        return _engine_cache
    
    if not settings.DATABASE_URL:
        raise ConfigurationError("DATABASE_URL no está configurada")
    
    try:
        engine = create_engine(
            settings.DATABASE_URL,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,  # Reciclar conexiones cada 30 minutos
            echo=False
        )
        
        # Probar conexión
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        logger.info("Conexión a PostgreSQL establecida exitosamente")
        _engine_cache = engine
        return engine
        
    except Exception as e:
        logger.error(f"Error conectando a PostgreSQL: {e}")
        raise ConnectionError(f"No se pudo conectar a la base de datos: {e}")


def close_database_engine() -> None:
    """Cierra el engine de base de datos y limpia el cache."""
    global _engine_cache
    
    if _engine_cache is not None:
        _engine_cache.dispose()
        _engine_cache = None
        logger.info("Conexiones a la base de datos cerradas")


def execute_query(query: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
    """
    Ejecuta una query SQL y retorna los resultados.
    
    Args:
        query: Query SQL a ejecutar.
        params: Parámetros para la query.
    
    Returns:
        List[Any]: Resultados de la query.
    """
    engine = get_database_engine()
    with engine.connect() as conn:
        result = conn.execute(text(query), params or {})
        return result.fetchall()


def table_exists(table_name: str) -> bool:
    """
    Verifica si una tabla existe en la base de datos.
    
    Args:
        table_name: Nombre de la tabla a verificar.
    
    Returns:
        bool: True si la tabla existe.
    """
    query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = :table_name
        );
    """
    result = execute_query(query, {'table_name': table_name})
    return result[0][0] if result else False


# =============================================================================
# Cliente HTTP Síncrono
# =============================================================================

def create_api_session() -> requests.Session:
    """
    Crea una sesión HTTP configurada para la API de Alegra.
    
    Returns:
        requests.Session: Sesión HTTP con headers de autenticación.
    
    Raises:
        ConfigurationError: Si ALEGRA_API_KEY no está configurada.
    """
    if not settings.ALEGRA_API_KEY:
        raise ConfigurationError("ALEGRA_API_KEY no está configurada")
    
    session = requests.Session()
    session.headers.update(settings.get_api_headers())
    return session


def safe_request(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    max_retries: Optional[int] = None,
    timeout: Optional[int] = None
) -> Any:
    """
    Realiza una petición HTTP con reintentos y manejo de errores.
    
    Args:
        session: Sesión HTTP a usar.
        url: URL a la cual hacer la petición.
        params: Parámetros de query string.
        max_retries: Número máximo de reintentos.
        timeout: Timeout en segundos.
    
    Returns:
        Any: Respuesta JSON de la API.
    
    Raises:
        RequestError: Si la petición falla después de todos los reintentos.
    """
    retries = max_retries or settings.MAX_RETRIES
    request_timeout = timeout or settings.REQUEST_TIMEOUT
    
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params, timeout=request_timeout)
            
            if response.status_code == 200:
                if not response.text.strip():
                    logger.warning("Respuesta vacía recibida")
                    return []
                return response.json()
            
            elif response.status_code == 429:
                # Rate limit - esperar y reintentar
                wait_time = settings.RETRY_DELAY_429
                logger.warning(
                    f"Rate limit (429). Esperando {wait_time}s... "
                    f"(Intento {attempt}/{retries})"
                )
                time.sleep(wait_time)
                continue
            
            else:
                logger.error(f"Error HTTP {response.status_code}: {response.text[:200]}")
                response.raise_for_status()
                
        except requests.exceptions.RequestException as e:
            if attempt == retries:
                raise RequestError(f"Error después de {retries} intentos: {e}")
            
            wait_time = settings.BACKOFF_FACTOR ** attempt
            logger.warning(f"Intento {attempt} falló ({e}). Reintentando en {wait_time:.1f}s...")
            time.sleep(wait_time)
    
    return []


class RequestError(Exception):
    """Excepción para errores de peticiones HTTP."""
    pass


# =============================================================================
# Cliente HTTP Asíncrono
# =============================================================================

async def fetch_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    max_retries: Optional[int] = None,
    timeout: Optional[int] = None
) -> Any:
    """
    Realiza una petición HTTP asíncrona con reintentos.
    
    Args:
        session: Sesión aiohttp a usar.
        url: URL a la cual hacer la petición.
        max_retries: Número máximo de reintentos.
        timeout: Timeout en segundos.
    
    Returns:
        Any: Respuesta JSON de la API, o lista vacía si falla.
    """
    retries = max_retries or settings.MAX_RETRIES
    request_timeout = timeout or settings.REQUEST_TIMEOUT
    
    for attempt in range(1, retries + 1):
        try:
            async with session.get(url, timeout=request_timeout) as response:
                if response.status == 200:
                    return await response.json()
                
                elif response.status == 429:
                    wait_time = settings.RETRY_DELAY_429
                    logger.warning(
                        f"⚠️ Rate limit (429) en {url}. "
                        f"Esperando {wait_time}s... (Intento {attempt}/{retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                
                else:
                    logger.error(f"❌ Error {response.status} en {url}")
                    return []
                    
        except asyncio.TimeoutError:
            logger.error(f"💥 Timeout en {url} (Intento {attempt}/{retries})")
            await asyncio.sleep(settings.NETWORK_ERROR_DELAY)
            
        except Exception as e:
            logger.error(
                f"💥 Error en {url}: {e}. "
                f"Reintentando en {settings.NETWORK_ERROR_DELAY}s... "
                f"(Intento {attempt}/{retries})"
            )
            await asyncio.sleep(settings.NETWORK_ERROR_DELAY)
    
    logger.error(f"⛔ Fallo definitivo en {url} tras {retries} intentos")
    return []


async def fetch_concurrent(
    urls: List[str],
    headers: Optional[Dict[str, str]] = None,
    concurrency: Optional[int] = None
) -> List[Any]:
    """
    Descarga múltiples URLs de manera concurrente.
    
    Args:
        urls: Lista de URLs a descargar.
        headers: Headers HTTP a usar.
        concurrency: Número máximo de peticiones simultáneas.
    
    Returns:
        List[Any]: Lista de respuestas JSON.
    """
    import nest_asyncio
    nest_asyncio.apply()
    
    max_concurrent = concurrency or settings.MAX_CONCURRENT_REQUESTS
    semaphore = asyncio.Semaphore(max_concurrent)
    request_headers = headers or settings.get_api_headers()
    
    async def bounded_fetch(url: str) -> Any:
        async with semaphore:
            return await fetch_with_retry(session, url)
    
    results = []
    async with aiohttp.ClientSession(headers=request_headers) as session:
        tasks = [bounded_fetch(url) for url in urls]
        results = await asyncio.gather(*tasks)
    
    return results


def run_async(coro):
    """
    Ejecuta una corutina de manera síncrona.
    
    Args:
        coro: Corutina a ejecutar.
    
    Returns:
        Resultado de la corutina.
    """
    import nest_asyncio
    nest_asyncio.apply()
    return asyncio.run(coro)


# =============================================================================
# Utilidades de Datos
# =============================================================================

def safe_numeric(value: Any, default: float = 0.0) -> float:
    """
    Convierte un valor a numérico de manera segura.
    
    Args:
        value: Valor a convertir.
        default: Valor por defecto si la conversión falla.
    
    Returns:
        float: Valor numérico.
    """
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Convierte un valor a entero de manera segura.
    
    Args:
        value: Valor a convertir.
        default: Valor por defecto si la conversión falla.
    
    Returns:
        int: Valor entero.
    """
    try:
        if value is None:
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default


def safe_string(value: Any, default: str = "") -> str:
    """
    Convierte un valor a string de manera segura.
    
    Args:
        value: Valor a convertir.
        default: Valor por defecto si la conversión falla.
    
    Returns:
        str: Valor como string.
    """
    if value is None:
        return default
    return str(value).strip()


# =============================================================================
# Decoradores
# =============================================================================

def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorador para reintentar una función en caso de error.
    
    Args:
        max_retries: Número máximo de reintentos.
        delay: Segundos a esperar entre reintentos.
        exceptions: Tupla de excepciones a capturar.
    
    Returns:
        Callable: Función decorada.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} falló (intento {attempt}/{max_retries}): {e}"
                        )
                        time.sleep(delay * attempt)
            raise last_exception
        return wrapper
    return decorator


def log_execution_time(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorador para loguear el tiempo de ejecución de una función.
    
    Args:
        func: Función a decorar.
    
    Returns:
        Callable: Función decorada.
    """
    def wrapper(*args, **kwargs) -> T:
        start_time = datetime.now()
        result = func(*args, **kwargs)
        duration = datetime.now() - start_time
        logger.info(f"{func.__name__} ejecutado en {duration}")
        return result
    return wrapper


if __name__ == "__main__":
    # Test de utilidades
    print("=== Test de Utilidades ===")
    
    # Test logging
    test_logger = setup_logging("test")
    test_logger.info("Test de logging OK")
    
    # Test conversiones seguras
    print(f"safe_numeric('123.45'): {safe_numeric('123.45')}")
    print(f"safe_numeric(None): {safe_numeric(None)}")
    print(f"safe_int('42'): {safe_int('42')}")
    print(f"safe_string(None): '{safe_string(None)}'")
    
    print("✅ Tests completados")

