#!/usr/bin/env python3
"""
Cron Runner para ejecutar main.py periódicamente
-------------------------------------------------

Este script está diseñado para ejecutarse como un servicio de cron.
Ejecuta main.py cada 3 días usando schedule.

Uso:
    python cron_runner.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from config import settings
from utils import setup_logging

# Intentar importar schedule
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False

# Configurar logging
logger = setup_logging("cron_runner")

# Configuración
INTERVAL_DAYS = settings.CRON_INTERVAL_DAYS
SCRIPT_NAME = "main.py"


def run_main_script() -> bool:
    """
    Ejecuta el script main.py.
    
    Returns:
        bool: True si la ejecución fue exitosa.
    """
    script_dir = Path(__file__).resolve().parent
    script_path = script_dir / SCRIPT_NAME
    
    if not script_path.exists():
        logger.error(f"❌ No se encontró el script {script_path}")
        return False
    
    logger.info(f"🚀 Ejecutando {SCRIPT_NAME}...")
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_dir),
            capture_output=False,
            text=True,
            check=True
        )
        
        duration = datetime.now() - start_time
        logger.info(f"✅ {SCRIPT_NAME} ejecutado exitosamente en {duration}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error ejecutando {SCRIPT_NAME}: código de salida {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        return False


def main():
    """Función principal del cron runner."""
    logger.info("="*60)
    logger.info(f"🔄 Iniciando Cron Runner - Intervalo: {INTERVAL_DAYS} días")
    logger.info("="*60)
    
    # Ejecutar inmediatamente al inicio
    logger.info("📅 Ejecutando primera ejecución...")
    run_main_script()
    
    # Calcular próximo intervalo
    next_run = datetime.now() + timedelta(days=INTERVAL_DAYS)
    logger.info(f"⏰ Próxima ejecución: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if SCHEDULE_AVAILABLE:
        # Usar schedule para programar ejecuciones
        schedule.every(INTERVAL_DAYS).days.do(run_main_script)
        
        logger.info("⏰ Servicio de cron iniciado. Esperando próximas ejecuciones...")
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Verificar cada hora
    else:
        # Fallback: usar time.sleep
        logger.warning("⚠️ Módulo 'schedule' no disponible. Usando modo simple.")
        while True:
            time.sleep(INTERVAL_DAYS * 24 * 60 * 60)
            logger.info("⏰ Ejecutando tarea programada...")
            run_main_script()
            next_run = datetime.now() + timedelta(days=INTERVAL_DAYS)
            logger.info(f"⏰ Próxima ejecución: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("🛑 Cron runner detenido por el usuario")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}")
        sys.exit(1)
