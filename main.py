#!/usr/bin/env python3
"""
combined_alegra_extract.py
==========================
Ejecución orquestada de los tres extractores de Alegra en un único comando.

Este script lanza, en el orden recomendado, los tres procesos que ya tienes:
  1. extractor_facturas_alegra.py   → facturas/ventas
  2. def_proveedor.py               → facturas de proveedor
  3. items-extract.py               → inventario (ítems)

Requisitos
----------
- Coloca este archivo **en la misma carpeta** que los otros tres scripts.
- Asegúrate de tener configuradas las mismas variables de entorno que utilizan
  los extractores (DATABASE_URL, ALEGRA_API_KEY, etc.).
- Instala las dependencias declaradas en cada script (psycopg2, SQLAlchemy,
  pandas, aiohttp, etc.).

Uso
---
```bash
python combined_alegra_extract.py        # ejecuta todo en secuencia
```

El script detiene la ejecución si alguno de los sub‑procesos devuelve un código
de error distinto de 0 o lanza una excepción no controlada.
"""

from __future__ import annotations

import logging
import pathlib
import runpy
import sys
from datetime import datetime

# Lista y orden de los scripts a ejecutar.  Ajusta las rutas si los has movido.
EXTRACTORS = [
    "extractor_facturas_alegra_sagrado.py",  # Ventas (facturas) - versión optimizada con concurrencia
    "extractor_facturas_proveedor_optimizado.py",  # Facturas de proveedor - versión optimizada
    "items-extract.py",              # Ítems / inventario
]


def run_script(path: pathlib.Path) -> None:
    """Ejecuta un script como si fuese `python <script>` y maneja errores."""
    logger = logging.getLogger(__name__)
    logger.info("=== Ejecutando %s ===", path.name)
    start_time = datetime.now()

    # runpy.run_path ejecuta el archivo manteniendo el intérprete actual.
    try:
        runpy.run_path(str(path), run_name="__main__")
    except SystemExit as exc:
        # Propagar cualquier código de salida distinto de 0
        code = exc.code if isinstance(exc.code, int) else 1
        if code != 0:
            logger.error("%s terminó con código de salida %s", path.name, code)
            sys.exit(code)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error al ejecutar %s: %s", path.name, exc)
        sys.exit(1)

    duration = datetime.now() - start_time
    logger.info("=== %s finalizado en %s ===", path.name, duration)


def main() -> None:
    """Punto de entrada principal."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Sobrescribe posibles configs previas
    )

    logger = logging.getLogger(__name__)
    script_dir = pathlib.Path(__file__).resolve().parent

    for script_name in EXTRACTORS:
        script_path = script_dir / script_name
        if not script_path.exists():
            logger.error("No se encontró el script %s", script_path)
            sys.exit(1)

        run_script(script_path)

    logger.info(">>> Proceso completo: facturas, facturas proveedor e ítems extraídos correctamente <<<")


if __name__ == "__main__":
    main()
