#!/usr/bin/env python3
"""
Tests para el módulo de utilidades.
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    safe_numeric,
    safe_int,
    safe_string,
    setup_logging,
    retry_on_failure,
)


class TestSafeConversions:
    """Tests para las funciones de conversión segura."""
    
    def test_safe_numeric_with_valid_number(self):
        """Test safe_numeric con número válido."""
        assert safe_numeric(123.45) == 123.45
        assert safe_numeric("123.45") == 123.45
        assert safe_numeric(100) == 100.0
    
    def test_safe_numeric_with_none(self):
        """Test safe_numeric con None."""
        assert safe_numeric(None) == 0.0
        assert safe_numeric(None, default=10.0) == 10.0
    
    def test_safe_numeric_with_invalid(self):
        """Test safe_numeric con valor inválido."""
        assert safe_numeric("abc") == 0.0
        assert safe_numeric("abc", default=5.0) == 5.0
        assert safe_numeric({}) == 0.0
        assert safe_numeric([]) == 0.0
    
    def test_safe_int_with_valid_number(self):
        """Test safe_int con número válido."""
        assert safe_int(42) == 42
        assert safe_int("42") == 42
        assert safe_int(42.9) == 42  # Trunca decimales
    
    def test_safe_int_with_none(self):
        """Test safe_int con None."""
        assert safe_int(None) == 0
        assert safe_int(None, default=10) == 10
    
    def test_safe_int_with_invalid(self):
        """Test safe_int con valor inválido."""
        assert safe_int("abc") == 0
        assert safe_int("abc", default=5) == 5
    
    def test_safe_string_with_valid_string(self):
        """Test safe_string con string válido."""
        assert safe_string("hello") == "hello"
        assert safe_string("  hello  ") == "hello"  # Trim
    
    def test_safe_string_with_none(self):
        """Test safe_string con None."""
        assert safe_string(None) == ""
        assert safe_string(None, default="N/A") == "N/A"
    
    def test_safe_string_with_number(self):
        """Test safe_string con número."""
        assert safe_string(123) == "123"
        assert safe_string(123.45) == "123.45"


class TestLogging:
    """Tests para la configuración de logging."""
    
    def test_setup_logging_creates_logger(self):
        """Test que setup_logging crea un logger."""
        logger = setup_logging("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"
    
    def test_setup_logging_same_logger_twice(self):
        """Test que setup_logging no duplica handlers."""
        logger1 = setup_logging("test_same")
        initial_handlers = len(logger1.handlers)
        
        logger2 = setup_logging("test_same")
        
        # No debe agregar más handlers
        assert len(logger2.handlers) == initial_handlers
        assert logger1 is logger2


class TestRetryDecorator:
    """Tests para el decorador retry_on_failure."""
    
    def test_retry_succeeds_first_try(self):
        """Test que no reintenta si la primera llamada tiene éxito."""
        call_count = 0
        
        @retry_on_failure(max_retries=3)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = successful_func()
        
        assert result == "success"
        assert call_count == 1
    
    def test_retry_succeeds_after_failures(self):
        """Test que reintenta y eventualmente tiene éxito."""
        call_count = 0
        
        @retry_on_failure(max_retries=3, delay=0.01)
        def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"
        
        result = eventually_successful()
        
        assert result == "success"
        assert call_count == 3
    
    def test_retry_fails_after_max_retries(self):
        """Test que falla después de agotar reintentos."""
        call_count = 0
        
        @retry_on_failure(max_retries=3, delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_fails()
        
        assert call_count == 3
    
    def test_retry_only_catches_specified_exceptions(self):
        """Test que solo captura las excepciones especificadas."""
        @retry_on_failure(max_retries=3, delay=0.01, exceptions=(ValueError,))
        def raises_type_error():
            raise TypeError("Wrong type")
        
        with pytest.raises(TypeError):
            raises_type_error()


class TestDatabaseFunctions:
    """Tests para funciones de base de datos (mockeadas)."""
    
    @patch('utils.settings')
    @patch('utils.create_engine')
    def test_get_database_engine_success(self, mock_create_engine, mock_settings):
        """Test que get_database_engine funciona con configuración válida."""
        mock_settings.DATABASE_URL = "postgresql://user:pass@localhost/db"
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        # Simular conexión exitosa
        mock_connection = MagicMock()
        mock_engine.connect.return_value.__enter__ = Mock(return_value=mock_connection)
        mock_engine.connect.return_value.__exit__ = Mock(return_value=False)
        
        from utils import get_database_engine, _engine_cache
        
        # Resetear cache
        import utils
        utils._engine_cache = None
        
        engine = get_database_engine()
        
        assert engine is not None
    
    @patch('utils.settings')
    def test_get_database_engine_no_url(self, mock_settings):
        """Test que get_database_engine falla sin DATABASE_URL."""
        mock_settings.DATABASE_URL = ""
        
        from utils import get_database_engine
        from config import ConfigurationError
        
        # Resetear cache
        import utils
        utils._engine_cache = None
        
        with pytest.raises(ConfigurationError):
            get_database_engine()


class TestAPIFunctions:
    """Tests para funciones de API (mockeadas)."""
    
    @patch('utils.settings')
    def test_create_api_session_success(self, mock_settings):
        """Test que create_api_session funciona con API key."""
        mock_settings.ALEGRA_API_KEY = "test_key"
        mock_settings.get_api_headers.return_value = {
            "accept": "application/json",
            "authorization": "Basic test_key"
        }
        
        from utils import create_api_session
        
        session = create_api_session()
        
        assert session is not None
        assert "authorization" in session.headers
    
    @patch('utils.settings')
    def test_create_api_session_no_key(self, mock_settings):
        """Test que create_api_session falla sin API key."""
        mock_settings.ALEGRA_API_KEY = ""
        
        from utils import create_api_session
        from config import ConfigurationError
        
        with pytest.raises(ConfigurationError):
            create_api_session()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

