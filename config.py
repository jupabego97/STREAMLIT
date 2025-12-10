#!/usr/bin/env python3
"""
Configuración Centralizada para la Aplicación STREAMLIT
--------------------------------------------------------

Este módulo contiene toda la configuración de la aplicación,
incluyendo variables de entorno, constantes y validaciones.

Uso:
    from config import settings
    
    # Acceder a configuración
    db_url = settings.DATABASE_URL
    api_key = settings.ALEGRA_API_KEY
    
    # Validar configuración (lanza error si falta algo crítico)
    settings.validate()
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()


class ConfigurationError(Exception):
    """Excepción para errores de configuración."""
    pass


@dataclass
class Settings:
    """Configuración centralizada de la aplicación."""
    
    # -------------------------------------------------------------------------
    # Base de Datos
    # -------------------------------------------------------------------------
    DATABASE_URL: str = field(default_factory=lambda: os.getenv('DATABASE_URL', ''))
    
    # Nombres de tablas
    TABLE_FACTURAS: str = "facturas"
    TABLE_FACTURAS_PROVEEDOR: str = "facturas_proveedor"
    TABLE_ITEMS: str = "items"
    TABLE_REPORTES: str = "reportes_ventas_30dias"
    
    # -------------------------------------------------------------------------
    # API de Alegra
    # -------------------------------------------------------------------------
    ALEGRA_API_KEY: str = field(default_factory=lambda: os.getenv('ALEGRA_API_KEY', ''))
    ALEGRA_BASE_URL: str = "https://api.alegra.com/api/v1"
    ALEGRA_INVOICES_URL: str = "https://api.alegra.com/api/v1/invoices"
    ALEGRA_BILLS_URL: str = "https://api.alegra.com/api/v1/bills"
    ALEGRA_ITEMS_URL: str = "https://api.alegra.com/api/v1/items"
    
    # -------------------------------------------------------------------------
    # Configuración de Requests/Concurrencia
    # -------------------------------------------------------------------------
    MAX_CONCURRENT_REQUESTS: int = 7
    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 5
    RETRY_DELAY_429: int = 60  # Segundos a esperar tras error 429 (rate limit)
    NETWORK_ERROR_DELAY: int = 5  # Segundos a esperar tras error de red
    BACKOFF_FACTOR: float = 1.5
    PAGE_SIZE: int = 30
    
    # -------------------------------------------------------------------------
    # Cache
    # -------------------------------------------------------------------------
    CACHE_TTL_SECONDS: int = 300  # 5 minutos
    
    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    LOG_LEVEL: int = logging.INFO
    LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    
    # -------------------------------------------------------------------------
    # Archivos de salida
    # -------------------------------------------------------------------------
    CSV_FACTURAS: str = "facturas_backup.csv"
    CSV_FACTURAS_PROVEEDOR: str = "facturas_proveedor.csv"
    CSV_ITEMS: str = "items.csv"
    EXPORT_TO_CSV: bool = True
    
    # -------------------------------------------------------------------------
    # Cron
    # -------------------------------------------------------------------------
    CRON_INTERVAL_DAYS: int = 3
    
    def validate(self) -> None:
        """
        Valida que todas las configuraciones críticas estén presentes.
        
        Raises:
            ConfigurationError: Si falta alguna configuración requerida.
        """
        errors = []
        
        if not self.DATABASE_URL:
            errors.append("DATABASE_URL es requerida. Configúrala en el archivo .env")
        
        if not self.ALEGRA_API_KEY:
            errors.append("ALEGRA_API_KEY es requerida. Configúrala en el archivo .env")
        
        if errors:
            error_msg = "Errores de configuración:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ConfigurationError(error_msg)
    
    def validate_for_dashboard(self) -> None:
        """
        Validación más ligera solo para el dashboard (no requiere API key).
        
        Raises:
            ConfigurationError: Si falta la configuración de base de datos.
        """
        if not self.DATABASE_URL:
            raise ConfigurationError(
                "DATABASE_URL es requerida para el dashboard. Configúrala en el archivo .env"
            )
    
    def get_api_headers(self) -> dict:
        """
        Retorna los headers necesarios para la API de Alegra.
        
        Returns:
            dict: Headers con autenticación.
        """
        return {
            "accept": "application/json",
            "authorization": f"Basic {self.ALEGRA_API_KEY}"
        }
    
    @property
    def is_production(self) -> bool:
        """Determina si estamos en producción basado en variables de entorno."""
        return os.getenv('ENVIRONMENT', 'development').lower() == 'production'
    
    def __repr__(self) -> str:
        """Representación segura (no muestra credenciales)."""
        return (
            f"Settings("
            f"DATABASE_URL={'***' if self.DATABASE_URL else 'NOT SET'}, "
            f"ALEGRA_API_KEY={'***' if self.ALEGRA_API_KEY else 'NOT SET'}, "
            f"TABLE_FACTURAS='{self.TABLE_FACTURAS}', "
            f"TABLE_REPORTES='{self.TABLE_REPORTES}')"
        )


# Instancia global de configuración
settings = Settings()


# -------------------------------------------------------------------------
# Funciones de utilidad para configuración
# -------------------------------------------------------------------------

def get_database_url() -> str:
    """
    Obtiene la URL de la base de datos.
    
    Returns:
        str: URL de conexión a PostgreSQL.
        
    Raises:
        ConfigurationError: Si no está configurada.
    """
    if not settings.DATABASE_URL:
        raise ConfigurationError("DATABASE_URL no está configurada")
    return settings.DATABASE_URL


def get_api_key() -> str:
    """
    Obtiene la API key de Alegra.
    
    Returns:
        str: API key de Alegra.
        
    Raises:
        ConfigurationError: Si no está configurada.
    """
    if not settings.ALEGRA_API_KEY:
        raise ConfigurationError("ALEGRA_API_KEY no está configurada")
    return settings.ALEGRA_API_KEY


def is_configured() -> bool:
    """
    Verifica si la aplicación está correctamente configurada.
    
    Returns:
        bool: True si todas las configuraciones críticas están presentes.
    """
    try:
        settings.validate()
        return True
    except ConfigurationError:
        return False


if __name__ == "__main__":
    # Test de configuración
    print("=== Test de Configuración ===")
    print(f"Settings: {settings}")
    
    try:
        settings.validate()
        print("✅ Configuración válida")
    except ConfigurationError as e:
        print(f"❌ {e}")

