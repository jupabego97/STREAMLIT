# Despliegue en Railway

Guía para desplegar la aplicación Streamlit de Reportes de Ventas en Railway.

## 📋 Requisitos Previos

1. Cuenta en [Railway](https://railway.app)
2. Base de datos PostgreSQL (puede ser un servicio de Railway o externa)

## 🚀 Pasos para Desplegar

### 1. Preparar el Repositorio

Asegúrate de tener estos archivos en tu directorio:
- `app_reporte_ventas.py` - Aplicación principal
- `requirements.txt` - Dependencias
- `Procfile` - Comando de inicio para Railway
- `railway.json` - Configuración de Railway (opcional)
- `.streamlit/config.toml` - Configuración de Streamlit

### 2. Crear un Nuevo Proyecto en Railway

1. Ve a [Railway Dashboard](https://railway.app/dashboard)
2. Haz clic en **"New Project"**
3. Selecciona **"Deploy from GitHub repo"** (si tu código está en GitHub) o **"Empty Project"**

### 3. Configurar Base de Datos PostgreSQL

Si no tienes una base de datos PostgreSQL:

1. En tu proyecto de Railway, haz clic en **"New"**
2. Selecciona **"Database"** → **"Add PostgreSQL"**
3. Railway creará automáticamente una base de datos PostgreSQL
4. Copia la **DATABASE_URL** desde las variables de entorno del servicio de PostgreSQL

### 4. Conectar el Repositorio (si usas GitHub)

1. Si aún no lo has hecho, sube tu código a un repositorio de GitHub
2. En Railway, haz clic en **"New"** → **"GitHub Repo"**
3. Selecciona tu repositorio y la rama (generalmente `main` o `master`)
4. Railway detectará automáticamente que es una aplicación Python

### 5. Configurar Variables de Entorno

1. En tu servicio de la aplicación en Railway, ve a la pestaña **"Variables"**
2. Agrega la variable de entorno:
   - **Nombre:** `DATABASE_URL`
   - **Valor:** La URL de conexión de PostgreSQL (copiada del paso 3)
     - Formato: `postgresql://usuario:contraseña@host:puerto/base_datos`

### 6. Configurar el Servicio

Railway debería detectar automáticamente:
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** Del `Procfile`: `streamlit run app_reporte_ventas.py --server.port=$PORT --server.address=0.0.0.0`

Si no se detecta automáticamente:
1. Ve a **Settings** → **Deploy**
2. En **Start Command**, asegúrate de que esté:
   ```
   streamlit run app_reporte_ventas.py --server.port=$PORT --server.address=0.0.0.0
   ```

### 7. Generar el Dominio Público

1. En tu servicio, ve a la pestaña **"Settings"**
2. Haz clic en **"Generate Domain"**
3. Railway generará una URL pública (ej: `tu-app.up.railway.app`)

### 8. Verificar el Despliegue

1. Railway desplegará automáticamente tu aplicación
2. Puedes ver los logs en tiempo real en la pestaña **"Deployments"**
3. Una vez completado, accede a la URL generada
4. La aplicación debería cargar mostrando el dashboard de reportes

## 🔧 Configuración Adicional

### Variables de Entorno Disponibles

- `DATABASE_URL`: URL de conexión a PostgreSQL (requerida)
- `PORT`: Puerto asignado por Railway (se configura automáticamente)

### Estructura de Archivos Requeridos

```
final/
├── app_reporte_ventas.py      # Aplicación Streamlit
├── requirements.txt            # Dependencias Python
├── Procfile                    # Comando de inicio
├── railway.json                # Configuración Railway (opcional)
├── runtime.txt                 # Versión de Python
└── .streamlit/
    └── config.toml            # Configuración Streamlit
```

## 📝 Notas Importantes

1. **Base de Datos:** Asegúrate de que la tabla `reportes_ventas_30dias` exista en tu base de datos PostgreSQL antes de usar la aplicación. Ejecuta primero `generar_reporte_ventas_30dias.py` o crea la tabla manualmente.

2. **Puerto:** Railway asigna automáticamente el puerto a través de la variable `$PORT`. El `Procfile` usa esta variable.

3. **Cache:** La aplicación usa cache de Streamlit, lo que puede requerir reiniciar el servicio si hay problemas.

4. **Logs:** Puedes ver los logs en tiempo real desde el dashboard de Railway.

## 🐛 Solución de Problemas

### La aplicación no inicia

- Verifica que `DATABASE_URL` esté configurada correctamente
- Revisa los logs en Railway para ver errores específicos
- Asegúrate de que todas las dependencias estén en `requirements.txt`

### Error de conexión a la base de datos

- Verifica que la `DATABASE_URL` tenga el formato correcto
- Asegúrate de que la base de datos PostgreSQL esté activa
- Verifica que la IP de Railway esté permitida en tu firewall (si usas BD externa)

### La página no carga

- Espera unos minutos después del despliegue
- Verifica que el servicio esté en estado "Running"
- Revisa los logs para errores de Streamlit

## 🔄 Actualizar la Aplicación

Para actualizar la aplicación después de cambios:

1. Si usas GitHub: Haz push de tus cambios y Railway desplegará automáticamente
2. Si no usas GitHub: Haz clic en **"Redeploy"** en Railway después de subir los cambios manualmente

## 📚 Recursos Adicionales

- [Documentación de Railway](https://docs.railway.app)
- [Documentación de Streamlit](https://docs.streamlit.io)
- [Guía de Railway para Python](https://docs.railway.app/guides/python)

