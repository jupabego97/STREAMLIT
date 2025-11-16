# Configuración de Cron Job en Railway

Guía para configurar la ejecución automática de `main.py` cada 3 días en Railway.

## 📋 Opciones de Configuración

Railway ofrece varias formas de ejecutar tareas programadas. Aquí te mostramos las mejores opciones:

## 🚀 Opción 1: Servicio Separado Continuo (Recomendado)

Esta es la forma más simple y confiable en Railway. El servicio ejecutará `main.py` inmediatamente al iniciar y luego cada 3 días.

### Paso 1: Crear un Nuevo Servicio en Railway

1. En tu proyecto de Railway, haz clic en **"New"**
2. Selecciona **"GitHub Repo"**
3. Selecciona el mismo repositorio `jupabego97/STREAMLIT`

### Paso 2: Configurar el Servicio Cron

1. En el nuevo servicio, ve a **"Settings"**
2. En **"Start Command"**, configura:
   ```
   python cron_runner.py
   ```
3. En **"Healthcheck"**, desactiva el healthcheck o déjalo en blanco (no es necesario para cron jobs)
4. En **"Restart Policy"**, selecciona **"ON_FAILURE"** para que se reinicie si falla

### Paso 3: Configurar Variables de Entorno

Asegúrate de que el servicio cron tenga las mismas variables de entorno que el servicio principal:
- `DATABASE_URL` - URL de conexión a PostgreSQL

**Nota:** Puedes compartir variables de entorno entre servicios en Railway usando **"Variables"** → **"New Variable"** y luego referenciarla en ambos servicios.

### Paso 4: Verificar el Funcionamiento

1. El servicio ejecutará `main.py` inmediatamente al iniciar
2. Luego esperará 3 días antes de la próxima ejecución
3. Puedes ver los logs en tiempo real en la pestaña **"Deployments"** del servicio

## 🔄 Opción 2: Usar Railway Scheduler (Alternativa)

Si Railway Cron no está disponible, puedes usar un servicio que se ejecute continuamente:

1. Crea un servicio separado
2. Usa el comando: `python cron_runner.py`
3. El script ejecutará `main.py` inmediatamente y luego esperará 3 días antes de la próxima ejecución

**Nota:** Esta opción mantiene el servicio corriendo todo el tiempo, lo que puede consumir más recursos.

## 📝 Opción 3: Usar GitHub Actions (Alternativa Externa)

Si prefieres no usar Railway para el cron, puedes configurar GitHub Actions:

1. Crea `.github/workflows/cron.yml`:
```yaml
name: Ejecutar Main cada 3 días

on:
  schedule:
    - cron: '0 0 */3 * *'  # Cada 3 días a medianoche UTC
  workflow_dispatch:  # Permite ejecución manual

jobs:
  run-main:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run main.py
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: python main.py
```

## ⚙️ Configuración Recomendada en Railway

### Estructura del Proyecto

```
Railway Project
├── Service 1: Streamlit App
│   ├── Start Command: streamlit run app_reporte_ventas.py --server.port=$PORT --server.address=0.0.0.0
│   └── Variables: DATABASE_URL
│
└── Service 2: Cron Job
    ├── Start Command: python cron_runner.py
    ├── Cron Schedule: 0 0 */3 * *
    └── Variables: DATABASE_URL
```

## 🔍 Verificar el Cron Job

1. Ve a los **logs** del servicio cron en Railway
2. Deberías ver mensajes como:
   ```
   🚀 Ejecutando main.py...
   ✅ main.py ejecutado exitosamente
   ```

## 📅 Horarios de Ejecución

El cron está configurado para ejecutarse cada 3 días a las 00:00 UTC.

**Ejemplos de ejecución:**
- 1 de enero 00:00 UTC
- 4 de enero 00:00 UTC
- 7 de enero 00:00 UTC
- 10 de enero 00:00 UTC
- etc.

## 🛠️ Solución de Problemas

### El cron no se ejecuta

1. Verifica que el servicio cron esté activo
2. Revisa los logs del servicio
3. Verifica que `DATABASE_URL` esté configurada
4. Asegúrate de que el cron schedule esté correctamente configurado

### Errores en la ejecución

1. Revisa los logs detallados en Railway
2. Verifica que todas las dependencias estén instaladas
3. Asegúrate de que la base de datos esté accesible

## 📚 Recursos

- [Railway Cron Documentation](https://docs.railway.app/guides/cron)
- [Cron Expression Guide](https://crontab.guru/)

