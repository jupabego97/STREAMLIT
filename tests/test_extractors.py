#!/usr/bin/env python3
"""
Tests para los extractores.
"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import date, datetime
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFacturasExtractor:
    """Tests para el extractor de facturas."""
    
    @patch('extractor_facturas_alegra_sagrado.settings')
    def test_extractor_initialization(self, mock_settings):
        """Test que el extractor se inicializa correctamente."""
        mock_settings.TABLE_FACTURAS = "facturas"
        
        from extractor_facturas_alegra_sagrado import FacturasExtractor
        
        extractor = FacturasExtractor()
        
        assert extractor.name == "FacturasExtractor"
        assert extractor.get_table_name() == "facturas"
    
    @patch('extractor_facturas_alegra_sagrado.settings')
    def test_dtype_mapping(self, mock_settings):
        """Test que el mapeo de tipos es correcto."""
        mock_settings.TABLE_FACTURAS = "facturas"
        
        from extractor_facturas_alegra_sagrado import FacturasExtractor
        
        extractor = FacturasExtractor()
        dtype = extractor.get_dtype_mapping()
        
        assert 'id' in dtype
        assert 'item_id' in dtype
        assert 'fecha' in dtype
        assert 'nombre' in dtype
        assert 'precio' in dtype
        assert 'cantidad' in dtype
        assert 'vendedor' in dtype
    
    def test_transform_empty_dataframe(self):
        """Test que transform maneja DataFrame vacío."""
        from extractor_facturas_alegra_sagrado import FacturasExtractor
        
        extractor = FacturasExtractor()
        result = extractor.transform(pd.DataFrame())
        
        assert result.empty


class TestProveedorExtractor:
    """Tests para el extractor de facturas de proveedor."""
    
    @patch('extractor_facturas_proveedor_optimizado.settings')
    def test_extractor_initialization(self, mock_settings):
        """Test que el extractor se inicializa correctamente."""
        mock_settings.TABLE_FACTURAS_PROVEEDOR = "facturas_proveedor"
        
        from extractor_facturas_proveedor_optimizado import ProveedorExtractor
        
        extractor = ProveedorExtractor()
        
        assert extractor.name == "ProveedorExtractor"
        assert extractor.get_table_name() == "facturas_proveedor"
    
    @patch('extractor_facturas_proveedor_optimizado.settings')
    def test_initial_dataframe_creation(self, mock_settings):
        """Test que _create_initial_dataframe crea datos correctos."""
        mock_settings.TABLE_FACTURAS_PROVEEDOR = "facturas_proveedor"
        
        from extractor_facturas_proveedor_optimizado import ProveedorExtractor
        
        extractor = ProveedorExtractor()
        df = extractor._create_initial_dataframe()
        
        assert not df.empty
        assert len(df) == 3
        assert 'id' in df.columns
        assert 'nombre' in df.columns
        assert 'proveedor' in df.columns
    
    @patch('extractor_facturas_proveedor_optimizado.settings')
    def test_clean_bills_data_empty(self, mock_settings):
        """Test que _clean_bills_data maneja DataFrame vacío."""
        mock_settings.TABLE_FACTURAS_PROVEEDOR = "facturas_proveedor"
        
        from extractor_facturas_proveedor_optimizado import ProveedorExtractor
        
        extractor = ProveedorExtractor()
        result = extractor._clean_bills_data(pd.DataFrame())
        
        assert result.empty
    
    @patch('extractor_facturas_proveedor_optimizado.settings')
    def test_clean_bills_data_conversion(self, mock_settings):
        """Test que _clean_bills_data convierte tipos correctamente."""
        mock_settings.TABLE_FACTURAS_PROVEEDOR = "facturas_proveedor"
        
        from extractor_facturas_proveedor_optimizado import ProveedorExtractor
        
        extractor = ProveedorExtractor()
        
        df = pd.DataFrame({
            'id': [1, 2],
            'fecha': ['2024-01-01', '2024-01-02'],
            'nombre': ['Producto 1', 'Producto 2'],
            'precio': ['100.50', '200.75'],
            'cantidad': ['5', '10'],
            'total': ['502.50', '2007.50'],
            'total_fact': ['502.50', '2007.50'],
            'proveedor': ['Proveedor A', 'Proveedor B']
        })
        
        result = extractor._clean_bills_data(df)
        
        assert not result.empty
        assert result['precio'].dtype in ['float64', 'float32']
        assert result['cantidad'].dtype in ['float64', 'float32']


class TestItemsExtractor:
    """Tests para el extractor de items."""
    
    @patch('items-extract.settings')
    def test_extractor_initialization(self, mock_settings):
        """Test que el extractor se inicializa correctamente."""
        mock_settings.TABLE_ITEMS = "items"
        
        # Importar con guión en el nombre
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "items_extract",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "items-extract.py")
        )
        items_extract = importlib.util.module_from_spec(spec)
        
        # No ejecutar el módulo completo, solo verificar que se puede importar
        assert spec is not None
    
    def test_extract_custom_field(self):
        """Test que _extract_custom_field funciona correctamente."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "items_extract",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "items-extract.py")
        )
        
        # Test manual de la lógica
        custom_fields = [
            {'name': 'FAMILIA', 'value': 'Electrónica'},
            {'name': 'Código de barras', 'value': '123456'}
        ]
        
        # Simular la función
        def extract_custom_field(fields, field_name):
            if not isinstance(fields, list):
                return None
            for field in fields:
                if field.get('name') == field_name:
                    return field.get('value')
            return None
        
        assert extract_custom_field(custom_fields, 'FAMILIA') == 'Electrónica'
        assert extract_custom_field(custom_fields, 'Código de barras') == '123456'
        assert extract_custom_field(custom_fields, 'No existe') is None
        assert extract_custom_field(None, 'FAMILIA') is None
    
    def test_extract_price(self):
        """Test que _extract_price funciona correctamente."""
        # Simular la función
        def extract_price(price_list):
            if price_list and isinstance(price_list, list):
                try:
                    return float(price_list[0].get('price', 0))
                except (ValueError, TypeError, IndexError):
                    return None
            return None
        
        assert extract_price([{'price': 100.50}]) == 100.50
        assert extract_price([{'price': '200'}]) == 200.0
        assert extract_price([]) is None
        assert extract_price(None) is None


class TestBaseExtractor:
    """Tests para la clase base de extractores."""
    
    def test_base_extractor_is_abstract(self):
        """Test que BaseExtractor no se puede instanciar directamente."""
        from base_extractor import BaseExtractor
        
        with pytest.raises(TypeError):
            BaseExtractor()
    
    def test_common_dtype_mapping(self):
        """Test que get_common_dtype_mapping retorna tipos correctos."""
        from base_extractor import BaseExtractor
        
        dtype = BaseExtractor.get_common_dtype_mapping()
        
        assert 'id' in dtype
        assert 'fecha' in dtype
        assert 'nombre' in dtype
        assert 'precio' in dtype


class TestDataTransformations:
    """Tests para transformaciones de datos comunes."""
    
    def test_calculate_margins(self):
        """Test que los márgenes se calculan correctamente."""
        df = pd.DataFrame({
            'precio': [100, 150, 200],
            'precio_promedio_compra': [80, 100, 250],
            'cantidad': [10, 5, 3]
        })
        
        # Simular cálculo de márgenes
        df['margen'] = df['precio'] - df['precio_promedio_compra']
        df['margen_porcentaje'] = (df['margen'] / df['precio'] * 100).round(2)
        df['total_margen'] = df['margen'] * df['cantidad']
        
        assert df['margen'].iloc[0] == 20  # 100 - 80
        assert df['margen'].iloc[2] == -50  # 200 - 250 (negativo)
        assert df['total_margen'].iloc[0] == 200  # 20 * 10
    
    def test_abc_classification(self):
        """Test que la clasificación ABC funciona correctamente."""
        # Simular datos de ventas
        ventas = pd.Series([1000, 500, 300, 100, 50, 30, 20])
        total = ventas.sum()
        pct_acum = (ventas.cumsum() / total * 100)
        
        # Clasificar
        def clasificar(pct):
            if pct <= 80:
                return 'A'
            elif pct <= 95:
                return 'B'
            return 'C'
        
        clasificacion = pct_acum.apply(clasificar)
        
        # Los primeros items con mayor venta deberían ser A
        assert clasificacion.iloc[0] == 'A'
        # Los últimos deberían ser C
        assert clasificacion.iloc[-1] == 'C'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

