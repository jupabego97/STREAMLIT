#!/usr/bin/env python3
"""
Orquestador Principal de Extractores de Alegra
-----------------------------------------------

Este script ejecuta todos los extractores y el generador de reportes
en el orden correcto:
  1. extractor_facturas_alegra_sagrado.py   → facturas/ventas
  2. extractor_facturas_proveedor_optimizado.py  → facturas de proveedor
  3. items-extract.py                       → inventario (ítems)
  4. generar_reporte_ventas_30dias.py       → reporte de ventas (SIEMPRE AL FINAL)

Uso:
    python main.py
"""
from __future__ import annotations

import logging
import pathlib
import runpy
import sys
from datetime import datetime

from config import settings, ConfigurationError
from utils import setup_logging

# Configurar logging
logger = setup_logging("main")

# Scripts a ejecutar en orden
EXTRACTORS = [
    "extractor_facturas_alegra_sagrado.py",
    "extractor_facturas_proveedor_optimizado.py",
    "items-extract.py",
    "generar_reporte_ventas_30dias.py",
]


def validate_configuration() -> bool:
    """Valida la configuración antes de ejecutar."""
    try:
        settings.validate()
        logger.info("✅ Configuración validada correctamente")
        return True
    except ConfigurationError as e:
        logger.error(f"❌ Error de configuración: {e}")
        return False


def run_script(path: pathlib.Path) -> bool:
    """
    Ejecuta un script y maneja errores.
    
    Args:
        path: Ruta al script a ejecutar.
    
    Returns:
        bool: True si el script se ejecutó correctamente.
    """
    logger.info(f"{'='*60}")
    logger.info(f"🚀 Ejecutando {path.name}")
    logger.info(f"{'='*60}")
    
    start_time = datetime.now()
    
    try:
        runpy.run_path(str(path), run_name="__main__")
        duration = datetime.now() - start_time
        logger.info(f"✅ {path.name} completado en {duration}")
        return True
        
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code != 0:
            logger.error(f"❌ {path.name} terminó con código {code}")
            return False
        duration = datetime.now() - start_time
        logger.info(f"✅ {path.name} completado en {duration}")
        return True
        
    except Exception as exc:
        logger.exception(f"❌ Error ejecutando {path.name}: {exc}")
        return False


def main() -> int:
    """
    Función principal del orquestador.
    
    Returns:
        int: Código de salida (0 = éxito, 1 = error).
    """
    logger.info("="*60)
    logger.info("🔄 Iniciando Orquestador de Extractores de Alegra")
    logger.info("="*60)
    
    start_time = datetime.now()
    
    # Validar configuración
    if not validate_configuration():
        logger.error("Abortando: Configuración inválida")
        return 1
    
    # Obtener directorio de scripts
    script_dir = pathlib.Path(__file__).resolve().parent
    
    # Ejecutar cada script en orden
    failed_scripts = []
    
    for script_name in EXTRACTORS:
        script_path = script_dir / script_name
        
        if not script_path.exists():
            logger.error(f"❌ No se encontró el script: {script_path}")
            failed_scripts.append(script_name)
            continue
        
        if not run_script(script_path):
            failed_scripts.append(script_name)
            # Continuar con los demás scripts aunque uno falle
            logger.warning(f"⚠️ Continuando a pesar del error en {script_name}")
    
    # Resumen final
    duration = datetime.now() - start_time
    logger.info("="*60)
    
    if failed_scripts:
        logger.warning(f"⚠️ Proceso completado con errores en: {', '.join(failed_scripts)}")
        logger.info(f"⏱️ Duración total: {duration}")
        return 1
    else:
        logger.info("✅ Todos los scripts ejecutados correctamente")
        logger.info(f"⏱️ Duración total: {duration}")
        return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🛑 Proceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"❌ Error fatal: {e}")
        sys.exit(1)
