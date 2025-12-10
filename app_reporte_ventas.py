#!/usr/bin/env python3
"""
Aplicación Streamlit para Reportes de Ventas de 30 Días
--------------------------------------------------------

Aplicación interactiva para visualizar, analizar y exportar datos
de la tabla reportes_ventas_30dias con:
- Dashboard con métricas y comparación con período anterior
- Sistema de alertas inteligentes
- Análisis de márgenes
- Predicciones de ventas
- Análisis ABC de productos
- Ranking de vendedores mejorado
- Filtros avanzados y rápidos
- Exportación a CSV/Excel
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

import streamlit as st

# Cargar variables de entorno
load_dotenv()

# Configuración de página
st.set_page_config(
    page_title="Reportes de Ventas - 30 Días",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
DB_URL_ENV = "DATABASE_URL"
TABLE_NAME = "reportes_ventas_30dias"


# =============================================================================
# Funciones de conexión y datos
# =============================================================================

@st.cache_resource
def get_database_engine():
    """Crea el engine de SQLAlchemy para PostgreSQL con cache."""
    db_url = os.getenv(DB_URL_ENV)
    if not db_url:
        st.error(f"⚠️ Variable {DB_URL_ENV} no encontrada. Configura la URL de la base de datos en .env")
        st.stop()
    
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"❌ Error conectando a PostgreSQL: {e}")
        st.stop()


@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    """Carga todos los datos de la tabla reportes_ventas_30dias."""
    try:
        engine = get_database_engine()
        query = f"SELECT * FROM {TABLE_NAME} ORDER BY fecha_venta DESC, nombre"
        df = pd.read_sql(query, engine)
        
        if not df.empty:
            df['fecha_venta'] = pd.to_datetime(df['fecha_venta'])
            df['precio'] = pd.to_numeric(df['precio'], errors='coerce')
            df['cantidad'] = pd.to_numeric(df['cantidad'], errors='coerce').astype('Int64')
            df['precio_promedio_compra'] = pd.to_numeric(df['precio_promedio_compra'], errors='coerce')
            
            # Calcular campos adicionales
            df['total_venta'] = df['precio'] * df['cantidad']
            df['margen'] = df['precio'] - df['precio_promedio_compra']
            df['margen_porcentaje'] = (df['margen'] / df['precio'] * 100).round(2)
            df['total_margen'] = df['margen'] * df['cantidad']
        
        return df
    except Exception as e:
        st.error(f"❌ Error cargando datos: {e}")
        st.stop()


@st.cache_data(ttl=300)
def load_previous_period_data() -> pd.DataFrame:
    """Carga datos del período anterior (30-60 días atrás) para comparación."""
    try:
        engine = get_database_engine()
        # Obtener datos de facturas del período anterior
        fecha_inicio = date.today() - timedelta(days=60)
        fecha_fin = date.today() - timedelta(days=31)
        
        query = f"""
            SELECT 
                f.nombre,
                f.precio,
                f.cantidad,
                f.metodo,
                f.vendedor,
                f.fecha as fecha_venta,
                i.familia,
                (f.precio * f.cantidad) as total_venta
            FROM facturas f
            LEFT JOIN items i ON f.item_id = i.id
            WHERE f.fecha BETWEEN '{fecha_inicio}' AND '{fecha_fin}'
        """
        
        df = pd.read_sql(query, engine)
        
        if not df.empty:
            df['fecha_venta'] = pd.to_datetime(df['fecha_venta'])
            df['precio'] = pd.to_numeric(df['precio'], errors='coerce')
            df['cantidad'] = pd.to_numeric(df['cantidad'], errors='coerce').astype('Int64')
            df['total_venta'] = pd.to_numeric(df['total_venta'], errors='coerce')
        
        return df
    except Exception:
        return pd.DataFrame()


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica filtros del sidebar al DataFrame."""
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Filtro de fechas
    fecha_inicio = st.session_state.get('fecha_inicio')
    fecha_fin = st.session_state.get('fecha_fin')
    if fecha_inicio and fecha_fin:
        mask = (filtered_df['fecha_venta'] >= pd.Timestamp(fecha_inicio)) & \
               (filtered_df['fecha_venta'] <= pd.Timestamp(fecha_fin))
        filtered_df = filtered_df[mask]
    
    # Filtro de productos
    productos = st.session_state.get('productos')
    if productos and len(productos) > 0:
        filtered_df = filtered_df[filtered_df['nombre'].isin(productos)]
    
    # Filtro de vendedores
    vendedores = st.session_state.get('vendedores')
    if vendedores and len(vendedores) > 0:
        filtered_df = filtered_df[filtered_df['vendedor'].isin(vendedores)]
    
    # Filtro de familias
    familias = st.session_state.get('familias')
    if familias and len(familias) > 0:
        filtered_df = filtered_df[filtered_df['familia'].isin(familias)]
    
    # Filtro de métodos
    metodos = st.session_state.get('metodos')
    if metodos and len(metodos) > 0:
        filtered_df = filtered_df[filtered_df['metodo'].isin(metodos)]
    
    # Filtro de proveedores
    proveedores = st.session_state.get('proveedores')
    if proveedores and len(proveedores) > 0:
        filtered_df = filtered_df[filtered_df['proveedor_moda'].isin(proveedores)]
    
    # Filtro de precios
    precio_range = st.session_state.get('precio_range')
    if precio_range and len(precio_range) == 2:
        mask = (filtered_df['precio'] >= precio_range[0]) & \
               (filtered_df['precio'] <= precio_range[1])
        filtered_df = filtered_df[mask]
    
    # Filtro de cantidades
    cantidad_range = st.session_state.get('cantidad_range')
    if cantidad_range and len(cantidad_range) == 2:
        mask = (filtered_df['cantidad'] >= cantidad_range[0]) & \
               (filtered_df['cantidad'] <= cantidad_range[1])
        filtered_df = filtered_df[mask]
    
    return filtered_df


# =============================================================================
# Sidebar - Filtros
# =============================================================================

def render_sidebar_filters(df: pd.DataFrame):
    """Renderiza los filtros en el sidebar."""
    st.sidebar.header("🔍 Filtros")
    
    # Filtros rápidos
    st.sidebar.subheader("⚡ Filtros Rápidos")
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        if st.button("Hoy", key="btn_hoy", use_container_width=True):
            st.session_state.fecha_inicio = date.today()
            st.session_state.fecha_fin = date.today()
            st.rerun()
    
    with col2:
        if st.button("7 días", key="btn_7dias", use_container_width=True):
            st.session_state.fecha_inicio = date.today() - timedelta(days=7)
            st.session_state.fecha_fin = date.today()
            st.rerun()
    
    with col3:
        if st.button("30 días", key="btn_30dias", use_container_width=True):
            st.session_state.fecha_inicio = date.today() - timedelta(days=30)
            st.session_state.fecha_fin = date.today()
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Valores por defecto
    fecha_min = df['fecha_venta'].min().date() if not df.empty else date.today() - timedelta(days=30)
    fecha_max = df['fecha_venta'].max().date() if not df.empty else date.today()
    
    # Rango de fechas
    st.sidebar.subheader("📅 Rango de Fechas")
    st.sidebar.date_input("Desde", value=fecha_min, key="fecha_inicio")
    st.sidebar.date_input("Hasta", value=fecha_max, key="fecha_fin")
    
    if not df.empty:
        # Productos
        productos_unicos = sorted(df['nombre'].dropna().unique())
        st.sidebar.subheader("📦 Productos")
        st.sidebar.multiselect("Seleccionar productos", options=productos_unicos, key="productos")
        
        # Vendedores
        vendedores_unicos = sorted(df['vendedor'].dropna().unique())
        st.sidebar.subheader("👤 Vendedores")
        st.sidebar.multiselect("Seleccionar vendedores", options=vendedores_unicos, key="vendedores")
        
        # Familias
        familias_unicas = sorted(df['familia'].dropna().unique())
        st.sidebar.subheader("🏷️ Familias")
        st.sidebar.multiselect("Seleccionar familias", options=familias_unicas, key="familias")
        
        # Métodos de pago
        metodos_unicos = sorted(df['metodo'].dropna().unique())
        st.sidebar.subheader("💳 Métodos de Pago")
        st.sidebar.multiselect("Seleccionar métodos", options=metodos_unicos, key="metodos")
        
        # Proveedores
        proveedores_unicos = sorted(df['proveedor_moda'].dropna().unique())
        st.sidebar.subheader("🏭 Proveedores")
        st.sidebar.multiselect("Seleccionar proveedores", options=proveedores_unicos, key="proveedores")
        
        # Rango de precios
        st.sidebar.subheader("💰 Rango de Precios")
        precio_min_val = float(df['precio'].min())
        precio_max_val = float(df['precio'].max())
        st.sidebar.slider(
            "Precio",
            min_value=precio_min_val,
            max_value=precio_max_val,
            value=(precio_min_val, precio_max_val),
            key="precio_range"
        )
        
        # Rango de cantidades
        st.sidebar.subheader("🔢 Rango de Cantidades")
        cantidad_min_val = int(df['cantidad'].min())
        cantidad_max_val = int(df['cantidad'].max())
        st.sidebar.slider(
            "Cantidad",
            min_value=cantidad_min_val,
            max_value=cantidad_max_val,
            value=(cantidad_min_val, cantidad_max_val),
            key="cantidad_range"
        )
    
    # Botón limpiar filtros
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Limpiar Filtros", use_container_width=True, key="limpiar_filtros"):
        for key in ['productos', 'vendedores', 'familias', 'metodos', 'proveedores']:
            if key in st.session_state:
                st.session_state[key] = []
        st.rerun()


# =============================================================================
# Sistema de Alertas
# =============================================================================

def render_alerts(df: pd.DataFrame):
    """Renderiza el sistema de alertas inteligentes."""
    if df.empty:
        return
    
    alerts = []
    
    # Alerta: Productos con margen negativo
    df_margen = df[df['precio_promedio_compra'].notna()]
    productos_negativos = df_margen[df_margen['margen'] < 0]
    if not productos_negativos.empty:
        count = len(productos_negativos)
        total_perdida = productos_negativos['total_margen'].sum()
        alerts.append({
            'type': 'error',
            'icon': '🚨',
            'title': f'{count} ventas con margen negativo',
            'detail': f'Pérdida total: ${abs(total_perdida):,.2f}',
            'data': productos_negativos[['nombre', 'precio', 'precio_promedio_compra', 'margen', 'cantidad']]
        })
    
    # Alerta: Productos con margen muy bajo (<10%)
    productos_margen_bajo = df_margen[(df_margen['margen'] > 0) & (df_margen['margen_porcentaje'] < 10)]
    if not productos_margen_bajo.empty:
        count = len(productos_margen_bajo)
        alerts.append({
            'type': 'warning',
            'icon': '⚠️',
            'title': f'{count} ventas con margen menor al 10%',
            'detail': 'Considera revisar los precios de estos productos',
            'data': productos_margen_bajo[['nombre', 'precio', 'margen_porcentaje', 'cantidad']].head(10)
        })
    
    # Alerta: Vendedores con bajo rendimiento (menos del 50% del promedio)
    if 'vendedor' in df.columns:
        ventas_por_vendedor = df.groupby('vendedor')['total_venta'].sum()
        promedio_ventas = ventas_por_vendedor.mean()
        vendedores_bajo = ventas_por_vendedor[ventas_por_vendedor < promedio_ventas * 0.5]
        if not vendedores_bajo.empty:
            alerts.append({
                'type': 'info',
                'icon': '📉',
                'title': f'{len(vendedores_bajo)} vendedores bajo el 50% del promedio',
                'detail': f'Promedio de ventas: ${promedio_ventas:,.2f}',
                'data': vendedores_bajo.reset_index()
            })
    
    # Mostrar alertas
    if alerts:
        st.subheader("🔔 Alertas del Sistema")
        for alert in alerts:
            if alert['type'] == 'error':
                with st.expander(f"{alert['icon']} {alert['title']}", expanded=True):
                    st.error(alert['detail'])
                    st.dataframe(alert['data'].head(10), use_container_width=True)
            elif alert['type'] == 'warning':
                with st.expander(f"{alert['icon']} {alert['title']}"):
                    st.warning(alert['detail'])
                    st.dataframe(alert['data'], use_container_width=True)
            else:
                with st.expander(f"{alert['icon']} {alert['title']}"):
                    st.info(alert['detail'])
                    st.dataframe(alert['data'], use_container_width=True)
        st.markdown("---")


# =============================================================================
# Dashboard - Métricas con Comparación
# =============================================================================

def calculate_delta(current: float, previous: float) -> Tuple[str, str]:
    """Calcula el delta porcentual entre dos valores."""
    if previous == 0:
        return "N/A", "off"
    
    delta = ((current - previous) / previous) * 100
    delta_str = f"{delta:+.1f}%"
    delta_color = "normal" if delta >= 0 else "inverse"
    return delta_str, delta_color


def render_metrics(df: pd.DataFrame, df_previous: pd.DataFrame):
    """Renderiza las métricas principales con comparación vs período anterior."""
    if df.empty:
        st.warning("⚠️ No hay datos para mostrar con los filtros aplicados.")
        return
    
    st.header("📊 Dashboard de Ventas")
    
    # Métricas actuales
    total_ventas = df['total_venta'].sum()
    total_registros = len(df)
    promedio_precio = df['precio'].mean()
    margen_promedio = df['margen'].mean() if 'margen' in df.columns else 0
    margen_total = df['total_margen'].sum() if 'total_margen' in df.columns else 0
    
    # Métricas período anterior
    prev_total_ventas = df_previous['total_venta'].sum() if not df_previous.empty else 0
    prev_registros = len(df_previous) if not df_previous.empty else 0
    prev_promedio = df_previous['precio'].mean() if not df_previous.empty else 0
    
    # Calcular deltas
    delta_ventas, _ = calculate_delta(total_ventas, prev_total_ventas)
    delta_registros, _ = calculate_delta(total_registros, prev_registros)
    delta_precio, _ = calculate_delta(promedio_precio, prev_promedio)
    
    # Mostrar métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Total Ventas",
            f"${total_ventas:,.2f}",
            delta=delta_ventas if prev_total_ventas > 0 else None,
            help="Comparado con los 30 días anteriores"
        )
    
    with col2:
        st.metric(
            "📝 Total Registros",
            f"{total_registros:,}",
            delta=delta_registros if prev_registros > 0 else None
        )
    
    with col3:
        st.metric(
            "📊 Precio Promedio",
            f"${promedio_precio:,.2f}",
            delta=delta_precio if prev_promedio > 0 else None
        )
    
    with col4:
        st.metric(
            "💵 Margen Total",
            f"${margen_total:,.2f}",
            delta=f"Promedio: ${margen_promedio:,.2f}"
        )
    
    st.markdown("---")
    
    # Top productos y vendedores
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 5 Productos Más Vendidos")
        top_productos = df.groupby('nombre').agg({
            'cantidad': 'sum',
            'total_venta': 'sum'
        }).sort_values('cantidad', ascending=False).head(5)
        
        if not top_productos.empty:
            st.dataframe(
                top_productos.style.format({
                    'cantidad': '{:,.0f}',
                    'total_venta': '${:,.2f}'
                }),
                use_container_width=True
            )
    
    with col2:
        st.subheader("👥 Top 5 Vendedores")
        top_vendedores = df.groupby('vendedor').agg({
            'total_venta': 'sum',
            'cantidad': 'sum'
        }).sort_values('total_venta', ascending=False).head(5)
        
        if not top_vendedores.empty:
            st.dataframe(
                top_vendedores.style.format({
                    'total_venta': '${:,.2f}',
                    'cantidad': '{:,.0f}'
                }),
                use_container_width=True
            )


# =============================================================================
# Gráficos
# =============================================================================

def render_charts(df: pd.DataFrame, key_prefix: str = ""):
    """Renderiza los gráficos interactivos."""
    if df.empty:
        return
    
    # Ventas por día
    st.subheader("📅 Ventas por Día")
    ventas_dia = df.groupby(df['fecha_venta'].dt.date).agg({
        'total_venta': 'sum',
        'cantidad': 'sum'
    }).reset_index()
    ventas_dia.columns = ['Fecha', 'Total Ventas', 'Cantidad']
    
    fig_line = px.line(
        ventas_dia,
        x='Fecha',
        y='Total Ventas',
        title='Evolución de Ventas Diarias',
        labels={'Total Ventas': 'Total ($)', 'Fecha': 'Fecha'}
    )
    fig_line.update_traces(line_color='#1f77b4', line_width=3)
    st.plotly_chart(fig_line, use_container_width=True, key=f"{key_prefix}chart_ventas_dia")
    
    # Gráficos en columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Ventas por Vendedor")
        ventas_vendedor = df.groupby('vendedor')['total_venta'].sum().sort_values(ascending=True).tail(10)
        fig_bar_vendedor = px.bar(
            x=ventas_vendedor.values,
            y=ventas_vendedor.index,
            orientation='h',
            title='Top 10 Vendedores',
            labels={'x': 'Total Ventas ($)', 'y': 'Vendedor'}
        )
        st.plotly_chart(fig_bar_vendedor, use_container_width=True, key=f"{key_prefix}chart_ventas_vendedor")
    
    with col2:
        st.subheader("🏷️ Ventas por Familia")
        ventas_familia = df.groupby('familia')['total_venta'].sum()
        fig_pie = px.pie(
            values=ventas_familia.values,
            names=ventas_familia.index,
            title='Distribución por Familia'
        )
        st.plotly_chart(fig_pie, use_container_width=True, key=f"{key_prefix}chart_ventas_familia")
    
    # Más gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💳 Ventas por Método de Pago")
        ventas_metodo = df.groupby('metodo')['total_venta'].sum().sort_values(ascending=False)
        fig_bar_metodo = px.bar(
            x=ventas_metodo.index,
            y=ventas_metodo.values,
            title='Ventas por Método',
            labels={'x': 'Método de Pago', 'y': 'Total Ventas ($)'}
        )
        st.plotly_chart(fig_bar_metodo, use_container_width=True, key=f"{key_prefix}chart_ventas_metodo")
    
    with col2:
        st.subheader("📦 Top 10 Productos por Cantidad")
        top_productos_cant = df.groupby('nombre')['cantidad'].sum().sort_values(ascending=False).head(10)
        fig_bar_productos = px.bar(
            x=top_productos_cant.index,
            y=top_productos_cant.values,
            title='Top 10 Productos',
            labels={'x': 'Producto', 'y': 'Cantidad Vendida'}
        )
        fig_bar_productos.update_xaxes(tickangle=45)
        st.plotly_chart(fig_bar_productos, use_container_width=True, key=f"{key_prefix}chart_top_productos")


# =============================================================================
# Análisis de Márgenes
# =============================================================================

def render_margin_analysis(df: pd.DataFrame):
    """Renderiza el análisis detallado de márgenes."""
    if df.empty:
        return
    
    st.header("💵 Análisis de Márgenes")
    
    df_margen = df[df['precio_promedio_compra'].notna()].copy()
    
    if df_margen.empty:
        st.info("ℹ️ No hay datos de precio de compra para analizar márgenes.")
        return
    
    # Métricas de margen
    col1, col2, col3, col4 = st.columns(4)
    
    margen_promedio = df_margen['margen'].mean()
    margen_total = df_margen['total_margen'].sum()
    productos_rentables = len(df_margen[df_margen['margen'] > 0])
    productos_no_rentables = len(df_margen[df_margen['margen'] <= 0])
    
    with col1:
        st.metric("💰 Margen Promedio", f"${margen_promedio:,.2f}")
    with col2:
        st.metric("💵 Margen Total", f"${margen_total:,.2f}")
    with col3:
        st.metric("✅ Ventas Rentables", f"{productos_rentables}")
    with col4:
        st.metric("⚠️ Ventas No Rentables", f"{productos_no_rentables}")
    
    st.markdown("---")
    
    # Gráfico scatter de márgenes
    st.subheader("📊 Precio de Venta vs Precio de Compra")
    fig_scatter = px.scatter(
        df_margen.head(200),
        x='precio_promedio_compra',
        y='precio',
        size='cantidad',
        color='margen_porcentaje',
        hover_data=['nombre', 'vendedor'],
        title='Análisis de Márgenes por Producto',
        labels={
            'precio_promedio_compra': 'Precio Compra ($)',
            'precio': 'Precio Venta ($)',
            'margen_porcentaje': 'Margen %'
        },
        color_continuous_scale='RdYlGn'
    )
    
    # Línea de referencia (margen cero)
    max_val = max(df_margen['precio'].max(), df_margen['precio_promedio_compra'].max())
    fig_scatter.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            name='Margen 0%',
            line=dict(dash='dash', color='gray')
        )
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Top productos por margen
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Productos por Margen Total")
        top_margen = df_margen.groupby('nombre').agg({
            'margen': 'mean',
            'total_margen': 'sum',
            'cantidad': 'sum'
        }).sort_values('total_margen', ascending=False).head(10)
        
        st.dataframe(
            top_margen.style.format({
                'margen': '${:,.2f}',
                'total_margen': '${:,.2f}',
                'cantidad': '{:,.0f}'
            }),
            use_container_width=True
        )
    
    with col2:
        st.subheader("📉 Productos con Menor Margen")
        bottom_margen = df_margen.groupby('nombre').agg({
            'margen': 'mean',
            'total_margen': 'sum',
            'cantidad': 'sum'
        }).sort_values('total_margen', ascending=True).head(10)
        
        st.dataframe(
            bottom_margen.style.format({
                'margen': '${:,.2f}',
                'total_margen': '${:,.2f}',
                'cantidad': '{:,.0f}'
            }),
            use_container_width=True
        )


# =============================================================================
# Predicciones de Ventas
# =============================================================================

def render_predictions(df: pd.DataFrame):
    """Renderiza las predicciones de ventas."""
    if df.empty:
        return
    
    st.header("🔮 Predicciones de Ventas")
    
    # Calcular ventas diarias
    ventas_dia = df.groupby(df['fecha_venta'].dt.date)['total_venta'].sum().reset_index()
    ventas_dia.columns = ['fecha', 'ventas']
    ventas_dia = ventas_dia.sort_values('fecha')
    
    if len(ventas_dia) < 7:
        st.warning("⚠️ Se necesitan al menos 7 días de datos para generar predicciones.")
        return
    
    # Calcular media móvil de 7 días
    ventas_dia['media_movil_7d'] = ventas_dia['ventas'].rolling(window=7, min_periods=1).mean()
    
    # Predicción simple basada en tendencia
    ultima_media = ventas_dia['media_movil_7d'].iloc[-1]
    tendencia = (ventas_dia['media_movil_7d'].iloc[-1] - ventas_dia['media_movil_7d'].iloc[-7]) / 7 if len(ventas_dia) >= 7 else 0
    
    # Métricas de predicción
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "📈 Venta Diaria Promedio (7d)",
            f"${ultima_media:,.2f}",
            delta=f"Tendencia: ${tendencia:+,.2f}/día"
        )
    
    with col2:
        prediccion_semanal = ultima_media * 7 + tendencia * 28  # 7 días + ajuste tendencia
        st.metric(
            "📅 Predicción Próxima Semana",
            f"${prediccion_semanal:,.2f}"
        )
    
    with col3:
        prediccion_mensual = ultima_media * 30 + tendencia * 465  # 30 días + ajuste
        st.metric(
            "📆 Predicción Próximo Mes",
            f"${prediccion_mensual:,.2f}"
        )
    
    st.markdown("---")
    
    # Gráfico de tendencia con predicción
    st.subheader("📊 Tendencia y Proyección")
    
    # Crear fechas futuras para predicción
    ultima_fecha = ventas_dia['fecha'].max()
    fechas_futuras = [ultima_fecha + timedelta(days=i) for i in range(1, 8)]
    predicciones = [ultima_media + tendencia * i for i in range(1, 8)]
    
    # Banda de confianza (±20%)
    predicciones_upper = [p * 1.2 for p in predicciones]
    predicciones_lower = [p * 0.8 for p in predicciones]
    
    fig = go.Figure()
    
    # Datos históricos
    fig.add_trace(go.Scatter(
        x=ventas_dia['fecha'],
        y=ventas_dia['ventas'],
        mode='lines+markers',
        name='Ventas Reales',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Media móvil
    fig.add_trace(go.Scatter(
        x=ventas_dia['fecha'],
        y=ventas_dia['media_movil_7d'],
        mode='lines',
        name='Media Móvil 7d',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    # Predicción
    fig.add_trace(go.Scatter(
        x=fechas_futuras,
        y=predicciones,
        mode='lines+markers',
        name='Predicción',
        line=dict(color='#2ca02c', width=2, dash='dot')
    ))
    
    # Banda de confianza
    fig.add_trace(go.Scatter(
        x=fechas_futuras + fechas_futuras[::-1],
        y=predicciones_upper + predicciones_lower[::-1],
        fill='toself',
        fillcolor='rgba(44, 160, 44, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Banda de Confianza (±20%)'
    ))
    
    fig.update_layout(
        title='Ventas Históricas y Proyección',
        xaxis_title='Fecha',
        yaxis_title='Ventas ($)',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de estacionalidad por día de la semana
    st.subheader("📅 Patrón por Día de la Semana")
    df['dia_semana'] = df['fecha_venta'].dt.day_name()
    ventas_dia_semana = df.groupby('dia_semana')['total_venta'].mean()
    
    # Ordenar días
    dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dias_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    ventas_dia_semana = ventas_dia_semana.reindex(dias_orden)
    
    fig_dias = px.bar(
        x=dias_es,
        y=ventas_dia_semana.values,
        title='Ventas Promedio por Día de la Semana',
        labels={'x': 'Día', 'y': 'Ventas Promedio ($)'}
    )
    st.plotly_chart(fig_dias, use_container_width=True)


# =============================================================================
# Análisis ABC
# =============================================================================

def render_abc_analysis(df: pd.DataFrame):
    """Renderiza el análisis ABC de productos."""
    if df.empty:
        return
    
    st.header("📊 Análisis ABC de Productos")
    
    st.markdown("""
    El análisis ABC clasifica los productos según su contribución a las ventas:
    - **Clase A**: Productos que representan el 80% de las ventas (los más importantes)
    - **Clase B**: Productos que representan el siguiente 15% de las ventas
    - **Clase C**: Productos que representan el 5% restante
    """)
    
    # Calcular ventas por producto
    ventas_producto = df.groupby('nombre').agg({
        'total_venta': 'sum',
        'cantidad': 'sum'
    }).sort_values('total_venta', ascending=False).reset_index()
    
    # Calcular porcentaje acumulado
    total_ventas = ventas_producto['total_venta'].sum()
    ventas_producto['porcentaje'] = (ventas_producto['total_venta'] / total_ventas * 100).round(2)
    ventas_producto['porcentaje_acumulado'] = ventas_producto['porcentaje'].cumsum()
    
    # Clasificar ABC
    def clasificar_abc(pct_acum):
        if pct_acum <= 80:
            return 'A'
        elif pct_acum <= 95:
            return 'B'
        return 'C'
    
    ventas_producto['clasificacion'] = ventas_producto['porcentaje_acumulado'].apply(clasificar_abc)
    
    # Métricas por clase
    col1, col2, col3 = st.columns(3)
    
    clase_a = ventas_producto[ventas_producto['clasificacion'] == 'A']
    clase_b = ventas_producto[ventas_producto['clasificacion'] == 'B']
    clase_c = ventas_producto[ventas_producto['clasificacion'] == 'C']
    
    with col1:
        st.metric(
            "🅰️ Clase A",
            f"{len(clase_a)} productos",
            delta=f"{clase_a['total_venta'].sum()/total_ventas*100:.1f}% de ventas"
        )
    
    with col2:
        st.metric(
            "🅱️ Clase B",
            f"{len(clase_b)} productos",
            delta=f"{clase_b['total_venta'].sum()/total_ventas*100:.1f}% de ventas"
        )
    
    with col3:
        st.metric(
            "©️ Clase C",
            f"{len(clase_c)} productos",
            delta=f"{clase_c['total_venta'].sum()/total_ventas*100:.1f}% de ventas"
        )
    
    st.markdown("---")
    
    # Gráfico de Pareto
    st.subheader("📈 Curva de Pareto")
    
    fig = go.Figure()
    
    # Barras de ventas
    fig.add_trace(go.Bar(
        x=list(range(1, len(ventas_producto) + 1)),
        y=ventas_producto['total_venta'],
        name='Ventas por Producto',
        marker_color=ventas_producto['clasificacion'].map({'A': '#2ca02c', 'B': '#ff7f0e', 'C': '#d62728'})
    ))
    
    # Línea de porcentaje acumulado
    fig.add_trace(go.Scatter(
        x=list(range(1, len(ventas_producto) + 1)),
        y=ventas_producto['porcentaje_acumulado'],
        mode='lines',
        name='% Acumulado',
        yaxis='y2',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Líneas de referencia 80% y 95%
    fig.add_hline(y=80, line_dash="dash", line_color="gray", annotation_text="80%", yref='y2')
    fig.add_hline(y=95, line_dash="dash", line_color="gray", annotation_text="95%", yref='y2')
    
    fig.update_layout(
        title='Análisis de Pareto - Curva ABC',
        xaxis_title='Productos (ordenados por ventas)',
        yaxis_title='Ventas ($)',
        yaxis2=dict(title='% Acumulado', overlaying='y', side='right', range=[0, 105]),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tablas por clase
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🅰️ Productos Clase A (Top)")
        st.dataframe(
            clase_a[['nombre', 'total_venta', 'cantidad', 'porcentaje']].head(20).style.format({
                'total_venta': '${:,.2f}',
                'cantidad': '{:,.0f}',
                'porcentaje': '{:.2f}%'
            }),
            use_container_width=True,
            height=400
        )
    
    with col2:
        st.subheader("📊 Resumen por Clase")
        resumen = ventas_producto.groupby('clasificacion').agg({
            'nombre': 'count',
            'total_venta': 'sum',
            'cantidad': 'sum'
        }).reset_index()
        resumen.columns = ['Clase', 'Productos', 'Ventas Totales', 'Cantidad Total']
        resumen['% Productos'] = (resumen['Productos'] / resumen['Productos'].sum() * 100).round(1)
        resumen['% Ventas'] = (resumen['Ventas Totales'] / resumen['Ventas Totales'].sum() * 100).round(1)
        
        st.dataframe(
            resumen.style.format({
                'Ventas Totales': '${:,.2f}',
                'Cantidad Total': '{:,.0f}',
                '% Productos': '{:.1f}%',
                '% Ventas': '{:.1f}%'
            }),
            use_container_width=True
        )


# =============================================================================
# Ranking de Vendedores Mejorado
# =============================================================================

def render_seller_ranking(df: pd.DataFrame):
    """Renderiza el ranking de vendedores mejorado."""
    if df.empty:
        return
    
    st.header("👤 Análisis por Vendedor")
    
    # Selector de vendedor
    vendedor_seleccionado = st.selectbox(
        "Seleccionar vendedor para análisis detallado",
        options=['Todos'] + sorted(df['vendedor'].dropna().unique().tolist())
    )
    
    # Calcular métricas por vendedor
    seller_stats = df.groupby('vendedor').agg({
        'total_venta': 'sum',
        'total_margen': 'sum',
        'nombre': 'nunique',
        'cantidad': 'sum',
        'precio': 'mean'
    }).reset_index()
    
    seller_stats.columns = ['Vendedor', 'Ventas Totales', 'Margen Total', 'Productos Únicos', 'Unidades', 'Ticket Promedio']
    seller_stats['Margen %'] = (seller_stats['Margen Total'] / seller_stats['Ventas Totales'] * 100).round(2)
    seller_stats = seller_stats.sort_values('Ventas Totales', ascending=False)
    
    # Promedios del equipo
    promedio_ventas = seller_stats['Ventas Totales'].mean()
    promedio_margen = seller_stats['Margen %'].mean()
    
    if vendedor_seleccionado == 'Todos':
        # Mostrar tabla de ranking
        st.subheader("🏆 Ranking de Vendedores")
        
        # Agregar indicador de rendimiento
        seller_stats['Rendimiento'] = seller_stats['Ventas Totales'].apply(
            lambda x: '🟢 Excelente' if x > promedio_ventas * 1.2 
            else ('🟡 Normal' if x > promedio_ventas * 0.8 else '🔴 Bajo')
        )
        
        st.dataframe(
            seller_stats.style.format({
                'Ventas Totales': '${:,.2f}',
                'Margen Total': '${:,.2f}',
                'Productos Únicos': '{:,.0f}',
                'Unidades': '{:,.0f}',
                'Ticket Promedio': '${:,.2f}',
                'Margen %': '{:.2f}%'
            }),
            use_container_width=True
        )
        
        # Gráfico de comparación
        st.subheader("📊 Comparación de Vendedores")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=seller_stats['Vendedor'],
            y=seller_stats['Ventas Totales'],
            name='Ventas Totales',
            marker_color='#1f77b4'
        ))
        
        fig.add_trace(go.Bar(
            x=seller_stats['Vendedor'],
            y=seller_stats['Margen Total'],
            name='Margen Total',
            marker_color='#2ca02c'
        ))
        
        # Línea de promedio
        fig.add_hline(y=promedio_ventas, line_dash="dash", line_color="red", 
                     annotation_text=f"Promedio: ${promedio_ventas:,.0f}")
        
        fig.update_layout(
            title='Ventas y Márgenes por Vendedor',
            xaxis_title='Vendedor',
            yaxis_title='Monto ($)',
            barmode='group',
            xaxis_tickangle=45
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Análisis individual del vendedor
        df_vendedor = df[df['vendedor'] == vendedor_seleccionado]
        stats_vendedor = seller_stats[seller_stats['Vendedor'] == vendedor_seleccionado].iloc[0]
        
        # Métricas del vendedor
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta = ((stats_vendedor['Ventas Totales'] - promedio_ventas) / promedio_ventas * 100)
            st.metric(
                "💰 Ventas Totales",
                f"${stats_vendedor['Ventas Totales']:,.2f}",
                delta=f"{delta:+.1f}% vs promedio"
            )
        
        with col2:
            st.metric("📦 Productos Únicos", f"{stats_vendedor['Productos Únicos']:,.0f}")
        
        with col3:
            st.metric("🎫 Ticket Promedio", f"${stats_vendedor['Ticket Promedio']:,.2f}")
        
        with col4:
            st.metric("📈 Margen %", f"{stats_vendedor['Margen %']:.2f}%")
        
        st.markdown("---")
        
        # Gráfico de ventas diarias del vendedor
        st.subheader(f"📅 Ventas Diarias de {vendedor_seleccionado}")
        ventas_vendedor_dia = df_vendedor.groupby(df_vendedor['fecha_venta'].dt.date)['total_venta'].sum()
        
        fig_vendedor = px.line(
            x=ventas_vendedor_dia.index,
            y=ventas_vendedor_dia.values,
            title=f'Evolución de Ventas',
            labels={'x': 'Fecha', 'y': 'Total Ventas ($)'}
        )
        fig_vendedor.update_traces(line_color='#1f77b4', line_width=3)
        st.plotly_chart(fig_vendedor, use_container_width=True)
        
        # Top productos del vendedor
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Productos")
            top_productos_vendedor = df_vendedor.groupby('nombre').agg({
                'total_venta': 'sum',
                'cantidad': 'sum'
            }).sort_values('total_venta', ascending=False).head(10)
            
            st.dataframe(
                top_productos_vendedor.style.format({
                    'total_venta': '${:,.2f}',
                    'cantidad': '{:,.0f}'
                }),
                use_container_width=True
            )
        
        with col2:
            st.subheader("💳 Métodos de Pago")
            metodos_vendedor = df_vendedor.groupby('metodo')['total_venta'].sum()
            fig_metodos = px.pie(
                values=metodos_vendedor.values,
                names=metodos_vendedor.index,
                title='Distribución por Método de Pago'
            )
            st.plotly_chart(fig_metodos, use_container_width=True)


# =============================================================================
# Tabla de Datos
# =============================================================================

def render_data_table(df: pd.DataFrame):
    """Renderiza la tabla interactiva de datos."""
    if df.empty:
        return
    
    st.header("📋 Tabla de Datos")
    
    # Búsqueda
    search_term = st.text_input("🔍 Buscar en todas las columnas", "")
    
    if search_term:
        mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
        df_filtered = df[mask]
    else:
        df_filtered = df.copy()
    
    # Seleccionar columnas
    default_cols = ['fecha_venta', 'nombre', 'precio', 'cantidad', 'total_venta', 
                   'vendedor', 'familia', 'metodo', 'proveedor_moda', 'precio_promedio_compra', 
                   'margen', 'margen_porcentaje']
    available_cols = [col for col in default_cols if col in df_filtered.columns]
    
    cols_selected = st.multiselect(
        "Seleccionar columnas a mostrar",
        options=available_cols,
        default=available_cols[:8],
        key="columnas_select"
    )
    
    if cols_selected:
        df_display = df_filtered[cols_selected].copy()
    else:
        df_display = df_filtered[available_cols].copy()
    
    # Formatear fechas
    if 'fecha_venta' in df_display.columns:
        df_display['fecha_venta'] = df_display['fecha_venta'].dt.strftime('%Y-%m-%d')
    
    # Mostrar tabla
    st.dataframe(
        df_display.style.format({
            'precio': '${:,.2f}',
            'cantidad': '{:,.0f}',
            'total_venta': '${:,.2f}',
            'precio_promedio_compra': '${:,.2f}',
            'margen': '${:,.2f}',
            'margen_porcentaje': '{:.2f}%'
        }, na_rep='N/A'),
        use_container_width=True,
        height=600
    )
    
    st.caption(f"Mostrando {len(df_display):,} de {len(df):,} registros")


# =============================================================================
# Exportación
# =============================================================================

def render_export(df: pd.DataFrame):
    """Renderiza la funcionalidad de exportación."""
    if df.empty:
        return
    
    st.header("💾 Exportar Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar CSV",
            data=csv,
            file_name=f"reporte_ventas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        try:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Ventas')
                
                resumen = pd.DataFrame({
                    'Métrica': ['Total Ventas', 'Total Registros', 'Precio Promedio', 
                               'Margen Promedio', 'Margen Total'],
                    'Valor': [
                        f"${df['total_venta'].sum():,.2f}",
                        len(df),
                        f"${df['precio'].mean():,.2f}",
                        f"${df['margen'].mean():,.2f}" if 'margen' in df.columns else 'N/A',
                        f"${df['total_margen'].sum():,.2f}" if 'total_margen' in df.columns else 'N/A'
                    ]
                })
                resumen.to_excel(writer, index=False, sheet_name='Resumen')
            
            excel_data = output.getvalue()
            st.download_button(
                label="📊 Descargar Excel",
                data=excel_data,
                file_name=f"reporte_ventas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except ImportError:
            st.error("⚠️ openpyxl no está instalado. Instala con: pip install openpyxl")


# =============================================================================
# Main App
# =============================================================================

def main():
    """Función principal de la aplicación."""
    # Título
    st.title("📊 Reportes de Ventas - Últimos 30 Días")
    st.markdown("""
    Dashboard interactivo para analizar ventas con:
    **Alertas** | **Predicciones** | **Análisis ABC** | **Ranking de Vendedores** | **Comparación de Períodos**
    """)
    
    # Cargar datos
    with st.spinner("Cargando datos..."):
        df = load_data()
        df_previous = load_previous_period_data()
    
    if df.empty:
        st.error("❌ No hay datos en la tabla. Ejecuta primero el script generar_reporte_ventas_30dias.py")
        st.stop()
    
    # Sidebar con filtros
    render_sidebar_filters(df)
    
    # Aplicar filtros
    df_filtered = apply_filters(df)
    
    # Sistema de alertas (siempre visible)
    render_alerts(df_filtered)
    
    # Tabs para organizar contenido
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Dashboard", 
        "💵 Márgenes",
        "🔮 Predicciones",
        "📈 Análisis ABC",
        "👤 Vendedores",
        "📋 Datos",
        "📈 Gráficos"
    ])
    
    with tab1:
        render_metrics(df_filtered, df_previous)
        render_charts(df_filtered, key_prefix="tab1_")
    
    with tab2:
        render_margin_analysis(df_filtered)
    
    with tab3:
        render_predictions(df_filtered)
    
    with tab4:
        render_abc_analysis(df_filtered)
    
    with tab5:
        render_seller_ranking(df_filtered)
    
    with tab6:
        render_data_table(df_filtered)
        st.markdown("---")
        render_export(df_filtered)
    
    with tab7:
        st.header("📈 Análisis Visual Completo")
        render_charts(df_filtered, key_prefix="tab7_")


if __name__ == "__main__":
    main()
