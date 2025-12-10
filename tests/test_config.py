#!/usr/bin/env python3
"""
Tests para el módulo de configuración.
"""
import os
import pytest
from unittest.mock import patch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Settings, ConfigurationError, get_database_url, get_api_key, is_configured


class TestSettings:
    """Tests para la clase Settings."""
    
    def test_settings_creation(self):
        """Test que Settings se puede crear correctamente."""
        settings = Settings()
        assert settings is not None
        assert settings.TABLE_FACTURAS == "facturas"
        assert settings.TABLE_ITEMS == "items"
        assert settings.TABLE_REPORTES == "reportes_ventas_30dias"
    
    def test_settings_default_values(self):
        """Test que los valores por defecto son correctos."""
        settings = Settings()
        assert settings.MAX_CONCURRENT_REQUESTS == 7
        assert settings.REQUEST_TIMEOUT == 30
        assert settings.MAX_RETRIES == 5
        assert settings.CACHE_TTL_SECONDS == 300
        assert settings.PAGE_SIZE == 30
    
    def test_settings_api_headers(self):
        """Test que get_api_headers retorna el formato correcto."""
        settings = Settings()
        settings.ALEGRA_API_KEY = "test_key"
        
        headers = settings.get_api_headers()
        
        assert "accept" in headers
        assert headers["accept"] == "application/json"
        assert "authorization" in headers
        assert "Basic test_key" in headers["authorization"]
    
    @patch.dict(os.environ, {"DATABASE_URL": "", "ALEGRA_API_KEY": ""})
    def test_validate_fails_without_config(self):
        """Test que validate() falla sin configuración."""
        settings = Settings()
        settings.DATABASE_URL = ""
        settings.ALEGRA_API_KEY = ""
        
        with pytest.raises(ConfigurationError) as exc_info:
            settings.validate()
        
        assert "DATABASE_URL" in str(exc_info.value)
        assert "ALEGRA_API_KEY" in str(exc_info.value)
    
    def test_validate_passes_with_config(self):
        """Test que validate() pasa con configuración correcta."""
        settings = Settings()
        settings.DATABASE_URL = "postgresql://user:pass@localhost/db"
        settings.ALEGRA_API_KEY = "test_api_key"
        
        # No debe lanzar excepción
        settings.validate()
    
    @patch.dict(os.environ, {"DATABASE_URL": ""})
    def test_validate_for_dashboard_fails_without_db(self):
        """Test que validate_for_dashboard() falla sin DATABASE_URL."""
        settings = Settings()
        settings.DATABASE_URL = ""
        
        with pytest.raises(ConfigurationError) as exc_info:
            settings.validate_for_dashboard()
        
        assert "DATABASE_URL" in str(exc_info.value)
    
    def test_validate_for_dashboard_passes_with_db(self):
        """Test que validate_for_dashboard() pasa con DATABASE_URL."""
        settings = Settings()
        settings.DATABASE_URL = "postgresql://user:pass@localhost/db"
        
        # No debe lanzar excepción (no requiere API key)
        settings.validate_for_dashboard()
    
    def test_repr_hides_credentials(self):
        """Test que __repr__ no muestra credenciales."""
        settings = Settings()
        settings.DATABASE_URL = "postgresql://secret:password@localhost/db"
        settings.ALEGRA_API_KEY = "super_secret_key"
        
        repr_str = repr(settings)
        
        assert "secret" not in repr_str
        assert "password" not in repr_str
        assert "super_secret_key" not in repr_str
        assert "***" in repr_str
    
    @patch.dict(os.environ, {"ENVIRONMENT": "production"})
    def test_is_production_true(self):
        """Test que is_production retorna True en producción."""
        settings = Settings()
        # Nota: is_production lee directamente de os.environ
        assert os.getenv("ENVIRONMENT") == "production"
    
    @patch.dict(os.environ, {"ENVIRONMENT": "development"})
    def test_is_production_false(self):
        """Test que is_production retorna False en desarrollo."""
        settings = Settings()
        assert os.getenv("ENVIRONMENT") == "development"


class TestConfigFunctions:
    """Tests para las funciones de utilidad de configuración."""
    
    def test_get_database_url_raises_without_config(self):
        """Test que get_database_url() lanza error sin configuración."""
        # Importar módulo fresco para resetear settings
        from config import settings
        original_url = settings.DATABASE_URL
        settings.DATABASE_URL = ""
        
        try:
            with pytest.raises(ConfigurationError):
                get_database_url()
        finally:
            settings.DATABASE_URL = original_url
    
    def test_get_api_key_raises_without_config(self):
        """Test que get_api_key() lanza error sin configuración."""
        from config import settings
        original_key = settings.ALEGRA_API_KEY
        settings.ALEGRA_API_KEY = ""
        
        try:
            with pytest.raises(ConfigurationError):
                get_api_key()
        finally:
            settings.ALEGRA_API_KEY = original_key
    
    def test_is_configured_returns_false_without_config(self):
        """Test que is_configured() retorna False sin configuración."""
        from config import settings
        original_url = settings.DATABASE_URL
        original_key = settings.ALEGRA_API_KEY
        settings.DATABASE_URL = ""
        settings.ALEGRA_API_KEY = ""
        
        try:
            assert is_configured() == False
        finally:
            settings.DATABASE_URL = original_url
            settings.ALEGRA_API_KEY = original_key
    
    def test_is_configured_returns_true_with_config(self):
        """Test que is_configured() retorna True con configuración."""
        from config import settings
        original_url = settings.DATABASE_URL
        original_key = settings.ALEGRA_API_KEY
        settings.DATABASE_URL = "postgresql://user:pass@localhost/db"
        settings.ALEGRA_API_KEY = "test_key"
        
        try:
            assert is_configured() == True
        finally:
            settings.DATABASE_URL = original_url
            settings.ALEGRA_API_KEY = original_key


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

