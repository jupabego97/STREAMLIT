#!/usr/bin/env python3
"""
Aplicación Streamlit para Reportes de Ventas de 30 Días
--------------------------------------------------------

Aplicación interactiva para visualizar, analizar y exportar datos
de la tabla reportes_ventas_30dias con dashboard, gráficos,
filtros avanzados y análisis de márgenes.
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from typing import Optional

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
    page_title="Sistema de Reportes y Pedidos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
DB_URL_ENV = "DATABASE_URL"
TABLE_NAME_VENTAS = "reportes_ventas_30dias"
TABLE_NAME_PEDIDOS = "para_pedidos"


# ---------------------------------------------------------------------------
# Funciones de conexión y datos
# ---------------------------------------------------------------------------

@st.cache_resource
def get_database_engine():
    """Crea el engine de SQLAlchemy para PostgreSQL con cache."""
    db_url = os.getenv(DB_URL_ENV)
    if not db_url:
        st.error(f"⚠️ Variable {DB_URL_ENV} no encontrada. Configura la URL de la base de datos en .env")
        st.stop()
    
    try:
        engine = create_engine(db_url)
        # Probar conexión
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"❌ Error conectando a PostgreSQL: {e}")
        st.stop()


@st.cache_data(ttl=300)  # Cache por 5 minutos
def load_data():
    """Carga todos los datos de la tabla reportes_ventas_30dias."""
    try:
        engine = get_database_engine()
        query = f"SELECT * FROM {TABLE_NAME_VENTAS} ORDER BY fecha_venta DESC, nombre"
        df = pd.read_sql(query, engine)
        
        # Convertir tipos de datos
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


# ---------------------------------------------------------------------------
# Funciones para Sección de Pedidos
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)  # Cache por 5 minutos
def load_pedidos_data():
    """Carga todos los datos de la tabla para_pedidos."""
    try:
        engine = get_database_engine()
        query = f"SELECT * FROM {TABLE_NAME_PEDIDOS} ORDER BY nombre"
        df = pd.read_sql(query, engine)
        
        # Convertir tipos de datos
        if not df.empty:
            numeric_cols = ['cantidad_disponible', 'precio_promedio_compra', 'cantidad_ultima_compra',
                           'precio_ultimo_compra', 'promedio_ventas_12_meses', 'cantidad_a_comprar',
                           'precio_promedio_venta', 'margen', 'utilidad']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            integer_cols = ['ventas_90_dias', 'ventas_60_dias', 'ventas_30_dias', 'ventas_15_dias', 'ventas_7_dias']
            for col in integer_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            
            if 'fecha_ultima_compra' in df.columns:
                df['fecha_ultima_compra'] = pd.to_datetime(df['fecha_ultima_compra'], errors='coerce')
            
            # Calcular campos adicionales
            df['valor_inventario'] = df['cantidad_disponible'] * df['precio_ultimo_compra']
            df['valor_pedido'] = df['cantidad_a_comprar'] * df['precio_ultimo_compra']
            df['dias_desde_ultima_compra'] = (pd.Timestamp.now() - df['fecha_ultima_compra']).dt.days
        
        return df
    except Exception as e:
        st.error(f"❌ Error cargando datos de pedidos: {e}")
        st.stop()


@st.cache_data(ttl=300)
def load_ventas_proveedor_30dias(proveedor: str):
    """Calcula ventas de un proveedor en los últimos 30 días desde facturas."""
    try:
        engine = get_database_engine()
        fecha_limite = (datetime.now() - timedelta(days=30)).date()
        
        query = text("""
            SELECT 
                COUNT(DISTINCT rv.nombre) as productos_unicos,
                SUM(rv.cantidad) as total_unidades,
                SUM(rv.precio * rv.cantidad) as total_valor
            FROM reportes_ventas_30dias rv
            WHERE rv.proveedor_moda = :proveedor
                AND rv.fecha_venta >= :fecha_limite
        """)
        
        result = pd.read_sql(query, engine, params={'proveedor': proveedor, 'fecha_limite': fecha_limite})
        
        if not result.empty:
            return {
                'productos_unicos': int(result.iloc[0]['productos_unicos'] or 0),
                'total_unidades': float(result.iloc[0]['total_unidades'] or 0),
                'total_valor': float(result.iloc[0]['total_valor'] or 0)
            }
        return {'productos_unicos': 0, 'total_unidades': 0, 'total_valor': 0}
    except Exception as e:
        st.warning(f"⚠️ Error calculando ventas del proveedor: {e}")
        return {'productos_unicos': 0, 'total_unidades': 0, 'total_valor': 0}


def apply_pedidos_filters(df: pd.DataFrame, proveedor: Optional[str] = None) -> pd.DataFrame:
    """Aplica filtros del sidebar al DataFrame de pedidos."""
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    # Filtro por proveedor
    if proveedor and proveedor != "Todos los proveedores":
        filtered_df = filtered_df[filtered_df['moda_proveedor'] == proveedor]
    
    # Filtro de productos
    productos = st.session_state.get('pedidos_productos')
    if productos and len(productos) > 0:
        filtered_df = filtered_df[filtered_df['nombre'].isin(productos)]
    
    # Filtro de familias
    familias = st.session_state.get('pedidos_familias')
    if familias and len(familias) > 0:
        filtered_df = filtered_df[filtered_df['familia'].isin(familias)]
    
    # Filtro de cantidad disponible
    cantidad_disp_range = st.session_state.get('pedidos_cantidad_disp_range')
    if cantidad_disp_range and len(cantidad_disp_range) == 2:
        mask = (filtered_df['cantidad_disponible'] >= cantidad_disp_range[0]) & \
               (filtered_df['cantidad_disponible'] <= cantidad_disp_range[1])
        filtered_df = filtered_df[mask]
    
    # Filtro de cantidad a comprar
    cantidad_comprar_range = st.session_state.get('pedidos_cantidad_comprar_range')
    if cantidad_comprar_range and len(cantidad_comprar_range) == 2:
        mask = (filtered_df['cantidad_a_comprar'] >= cantidad_comprar_range[0]) & \
               (filtered_df['cantidad_a_comprar'] <= cantidad_comprar_range[1])
        filtered_df = filtered_df[mask]
    
    # Filtro de ventas 7 días
    ventas_7_range = st.session_state.get('pedidos_ventas_7_range')
    if ventas_7_range and len(ventas_7_range) == 2:
        mask = (filtered_df['ventas_7_dias'] >= ventas_7_range[0]) & \
               (filtered_df['ventas_7_dias'] <= ventas_7_range[1])
        filtered_df = filtered_df[mask]
    
    # Filtro de ventas 30 días
    ventas_30_range = st.session_state.get('pedidos_ventas_30_range')
    if ventas_30_range and len(ventas_30_range) == 2:
        mask = (filtered_df['ventas_30_dias'] >= ventas_30_range[0]) & \
               (filtered_df['ventas_30_dias'] <= ventas_30_range[1])
        filtered_df = filtered_df[mask]
    
    # Filtro de ventas 90 días
    ventas_90_range = st.session_state.get('pedidos_ventas_90_range')
    if ventas_90_range and len(ventas_90_range) == 2:
        mask = (filtered_df['ventas_90_dias'] >= ventas_90_range[0]) & \
               (filtered_df['ventas_90_dias'] <= ventas_90_range[1])
        filtered_df = filtered_df[mask]
    
    # Filtro de margen
    margen_range = st.session_state.get('pedidos_margen_range')
    if margen_range and len(margen_range) == 2:
        mask = (filtered_df['margen'] >= margen_range[0]) & \
               (filtered_df['margen'] <= margen_range[1])
        filtered_df = filtered_df[mask]
    
    # Filtro de utilidad
    utilidad_range = st.session_state.get('pedidos_utilidad_range')
    if utilidad_range and len(utilidad_range) == 2:
        mask = (filtered_df['utilidad'] >= utilidad_range[0]) & \
               (filtered_df['utilidad'] <= utilidad_range[1])
        filtered_df = filtered_df[mask]
    
    # Filtro de precio promedio compra
    precio_compra_range = st.session_state.get('pedidos_precio_compra_range')
    if precio_compra_range and len(precio_compra_range) == 2:
        mask = (filtered_df['precio_promedio_compra'] >= precio_compra_range[0]) & \
               (filtered_df['precio_promedio_compra'] <= precio_compra_range[1])
        filtered_df = filtered_df[mask]
    
    # Filtro de precio promedio venta
    precio_venta_range = st.session_state.get('pedidos_precio_venta_range')
    if precio_venta_range and len(precio_venta_range) == 2:
        mask = (filtered_df['precio_promedio_venta'] >= precio_venta_range[0]) & \
               (filtered_df['precio_promedio_venta'] <= precio_venta_range[1])
        filtered_df = filtered_df[mask]
    
    # Filtro de días desde última compra
    dias_compra_range = st.session_state.get('pedidos_dias_compra_range')
    if dias_compra_range and len(dias_compra_range) == 2:
        mask = (filtered_df['dias_desde_ultima_compra'] >= dias_compra_range[0]) & \
               (filtered_df['dias_desde_ultima_compra'] <= dias_compra_range[1])
        filtered_df = filtered_df[mask]
    
    # Filtro de fecha última compra
    fecha_compra_inicio = st.session_state.get('pedidos_fecha_compra_inicio')
    fecha_compra_fin = st.session_state.get('pedidos_fecha_compra_fin')
    if fecha_compra_inicio and fecha_compra_fin:
        mask = (filtered_df['fecha_ultima_compra'] >= pd.Timestamp(fecha_compra_inicio)) & \
               (filtered_df['fecha_ultima_compra'] <= pd.Timestamp(fecha_compra_fin))
        filtered_df = filtered_df[mask]
    
    # Filtro de solo productos a comprar
    solo_comprar = st.session_state.get('pedidos_solo_comprar', False)
    if solo_comprar:
        filtered_df = filtered_df[filtered_df['cantidad_a_comprar'] > 0]
    
    # Filtro de stock bajo
    stock_bajo = st.session_state.get('pedidos_stock_bajo', False)
    umbral_stock = st.session_state.get('pedidos_umbral_stock', 10)
    if stock_bajo:
        filtered_df = filtered_df[filtered_df['cantidad_disponible'] < umbral_stock]
    
    # Ordenamiento
    ordenar_por = st.session_state.get('pedidos_ordenar_por', 'cantidad_a_comprar')
    orden_desc = st.session_state.get('pedidos_orden_desc', True)
    if ordenar_por in filtered_df.columns:
        filtered_df = filtered_df.sort_values(ordenar_por, ascending=not orden_desc)
    
    return filtered_df


# ---------------------------------------------------------------------------
# Sidebar - Filtros
# ---------------------------------------------------------------------------

def render_sidebar_filters(df: pd.DataFrame):
    """Renderiza los filtros en el sidebar."""
    st.sidebar.header("🔍 Filtros")
    
    # Inicializar valores por defecto
    fecha_min = df['fecha_venta'].min().date() if not df.empty else date.today() - timedelta(days=30)
    fecha_max = df['fecha_venta'].max().date() if not df.empty else date.today()
    
    # Rango de fechas
    st.sidebar.subheader("📅 Rango de Fechas")
    st.sidebar.date_input(
        "Desde",
        value=fecha_min,
        key="fecha_inicio"
    )
    st.sidebar.date_input(
        "Hasta",
        value=fecha_max,
        key="fecha_fin"
    )
    
    if not df.empty:
        # Productos
        productos_unicos = sorted(df['nombre'].dropna().unique())
        st.sidebar.subheader("📦 Productos")
        st.sidebar.multiselect(
            "Seleccionar productos",
            options=productos_unicos,
            key="productos"
        )
        
        # Vendedores
        vendedores_unicos = sorted(df['vendedor'].dropna().unique())
        st.sidebar.subheader("👤 Vendedores")
        st.sidebar.multiselect(
            "Seleccionar vendedores",
            options=vendedores_unicos,
            key="vendedores"
        )
        
        # Familias
        familias_unicas = sorted(df['familia'].dropna().unique())
        st.sidebar.subheader("🏷️ Familias")
        st.sidebar.multiselect(
            "Seleccionar familias",
            options=familias_unicas,
            key="familias"
        )
        
        # Métodos de pago
        metodos_unicos = sorted(df['metodo'].dropna().unique())
        st.sidebar.subheader("💳 Métodos de Pago")
        st.sidebar.multiselect(
            "Seleccionar métodos",
            options=metodos_unicos,
            key="metodos"
        )
        
        # Proveedores
        proveedores_unicos = sorted(df['proveedor_moda'].dropna().unique())
        st.sidebar.subheader("🏭 Proveedores")
        st.sidebar.multiselect(
            "Seleccionar proveedores",
            options=proveedores_unicos,
            key="proveedores"
        )
        
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
        # Los widgets se limpiarán automáticamente al hacer rerun porque usan valores por defecto
        st.rerun()


# ---------------------------------------------------------------------------
# Dashboard - Métricas
# ---------------------------------------------------------------------------

def render_metrics(df: pd.DataFrame):
    """Renderiza las métricas principales del dashboard."""
    if df.empty:
        st.warning("⚠️ No hay datos para mostrar con los filtros aplicados.")
        return
    
    st.header("📊 Dashboard de Ventas")
    
    # Calcular métricas
    total_ventas = df['total_venta'].sum()
    total_registros = len(df)
    promedio_precio = df['precio'].mean()
    margen_promedio = df['margen'].mean()
    margen_total = df['total_margen'].sum()
    
    # Mostrar métricas en columnas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Total Ventas",
            f"${total_ventas:,.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "📝 Total Registros",
            f"{total_registros:,}",
            delta=None
        )
    
    with col3:
        st.metric(
            "📊 Precio Promedio",
            f"${promedio_precio:,.2f}",
            delta=None
        )
    
    with col4:
        st.metric(
            "💵 Margen Promedio",
            f"${margen_promedio:,.2f}",
            delta=f"Total: ${margen_total:,.2f}"
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


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def render_charts(df: pd.DataFrame, key_prefix: str = ""):
    """Renderiza los gráficos interactivos."""
    if df.empty:
        return
    
    # Mostrar header solo si hay prefijo (llamado desde tab3)
    if key_prefix:
        st.header("📈 Análisis Visual")
    
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
    
    # Análisis de márgenes
    st.subheader("💵 Análisis de Márgenes (Precio Venta vs Compra)")
    df_margen = df[df['precio_promedio_compra'].notna()].copy()
    if not df_margen.empty:
        fig_scatter = px.scatter(
            df_margen.head(100),  # Limitar para mejor rendimiento
            x='precio_promedio_compra',
            y='precio',
            size='cantidad',
            color='margen_porcentaje',
            hover_data=['nombre', 'vendedor'],
            title='Precio de Venta vs Precio de Compra Promedio',
            labels={
                'precio_promedio_compra': 'Precio Compra Promedio ($)',
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
        st.plotly_chart(fig_scatter, use_container_width=True, key=f"{key_prefix}chart_margen_scatter")
    
    # Gráficos de barras combinados con línea de utilidad
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏷️ Ventas y Márgenes por Familia")
        df_familia = df[df['familia'].notna()].copy()
        if not df_familia.empty:
            familia_metrics = df_familia.groupby('familia').agg({
                'total_venta': 'sum',
                'total_margen': 'sum'
            }).reset_index()
            
            # Calcular porcentaje de utilidad
            familia_metrics['utilidad_porcentaje'] = (
                (familia_metrics['total_margen'] / familia_metrics['total_venta']) * 100
            ).round(2)
            
            # Ordenar por ventas totales (descendente) y tomar top 10
            familia_metrics = familia_metrics.sort_values('total_venta', ascending=False).head(10)
            
            # Crear gráfico combinado
            fig_familia = go.Figure()
            
            # Barra 1: Ventas Totales
            fig_familia.add_trace(go.Bar(
                x=familia_metrics['familia'],
                y=familia_metrics['total_venta'],
                name='Ventas Totales',
                yaxis='y',
                marker_color='#1f77b4',
                text=[f'${val:,.0f}' for val in familia_metrics['total_venta']],
                textposition='outside'
            ))
            
            # Barra 2: Margen Total
            fig_familia.add_trace(go.Bar(
                x=familia_metrics['familia'],
                y=familia_metrics['total_margen'],
                name='Margen Total',
                yaxis='y',
                marker_color='#2ca02c',
                text=[f'${val:,.0f}' for val in familia_metrics['total_margen']],
                textposition='outside'
            ))
            
            # Línea: Porcentaje de Utilidad
            fig_familia.add_trace(go.Scatter(
                x=familia_metrics['familia'],
                y=familia_metrics['utilidad_porcentaje'],
                name='% Utilidad',
                yaxis='y2',
                mode='lines+markers',
                line=dict(color='#ff7f0e', width=3),
                marker=dict(size=8),
                text=[f'{val:.1f}%' for val in familia_metrics['utilidad_porcentaje']],
                textposition='top center'
            ))
            
            fig_familia.update_layout(
                xaxis=dict(title='Familia', tickangle=45),
                yaxis=dict(title='Monto ($)', side='left'),
                yaxis2=dict(title='% Utilidad', overlaying='y', side='right', range=[0, familia_metrics['utilidad_porcentaje'].max() * 1.2]),
                barmode='group',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig_familia, use_container_width=True, key=f"{key_prefix}chart_familia_ventas_margen")
    
    with col2:
        st.subheader("🏭 Ventas y Márgenes por Proveedor")
        df_proveedor = df[df['proveedor_moda'].notna()].copy()
        if not df_proveedor.empty:
            proveedor_metrics = df_proveedor.groupby('proveedor_moda').agg({
                'total_venta': 'sum',
                'total_margen': 'sum'
            }).reset_index()
            
            # Calcular porcentaje de utilidad
            proveedor_metrics['utilidad_porcentaje'] = (
                (proveedor_metrics['total_margen'] / proveedor_metrics['total_venta']) * 100
            ).round(2)
            
            # Ordenar por ventas totales (descendente) y tomar top 10
            proveedor_metrics = proveedor_metrics.sort_values('total_venta', ascending=False).head(10)
            
            # Crear gráfico combinado
            fig_proveedor = go.Figure()
            
            # Barra 1: Ventas Totales
            fig_proveedor.add_trace(go.Bar(
                x=proveedor_metrics['proveedor_moda'],
                y=proveedor_metrics['total_venta'],
                name='Ventas Totales',
                yaxis='y',
                marker_color='#1f77b4',
                text=[f'${val:,.0f}' for val in proveedor_metrics['total_venta']],
                textposition='outside'
            ))
            
            # Barra 2: Margen Total
            fig_proveedor.add_trace(go.Bar(
                x=proveedor_metrics['proveedor_moda'],
                y=proveedor_metrics['total_margen'],
                name='Margen Total',
                yaxis='y',
                marker_color='#2ca02c',
                text=[f'${val:,.0f}' for val in proveedor_metrics['total_margen']],
                textposition='outside'
            ))
            
            # Línea: Porcentaje de Utilidad
            fig_proveedor.add_trace(go.Scatter(
                x=proveedor_metrics['proveedor_moda'],
                y=proveedor_metrics['utilidad_porcentaje'],
                name='% Utilidad',
                yaxis='y2',
                mode='lines+markers',
                line=dict(color='#ff7f0e', width=3),
                marker=dict(size=8),
                text=[f'{val:.1f}%' for val in proveedor_metrics['utilidad_porcentaje']],
                textposition='top center'
            ))
            
            fig_proveedor.update_layout(
                xaxis=dict(title='Proveedor', tickangle=45),
                yaxis=dict(title='Monto ($)', side='left'),
                yaxis2=dict(title='% Utilidad', overlaying='y', side='right', range=[0, proveedor_metrics['utilidad_porcentaje'].max() * 1.2]),
                barmode='group',
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig_proveedor, use_container_width=True, key=f"{key_prefix}chart_proveedor_ventas_margen")


# ---------------------------------------------------------------------------
# Análisis de Márgenes
# ---------------------------------------------------------------------------

def render_margin_analysis(df: pd.DataFrame):
    """Renderiza el análisis detallado de márgenes."""
    if df.empty:
        return
    
    st.header("💵 Análisis de Márgenes")
    
    # Filtrar solo productos con precio de compra
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
        st.metric("✅ Productos Rentables", f"{productos_rentables}")
    with col4:
        st.metric("⚠️ Productos No Rentables", f"{productos_no_rentables}")
    
    # Alertas
    productos_margen_negativo = df_margen[df_margen['margen'] < 0]
    if not productos_margen_negativo.empty:
        st.warning(f"⚠️ Hay {len(productos_margen_negativo)} productos con margen negativo")
    
    productos_margen_bajo = df_margen[(df_margen['margen'] > 0) & (df_margen['margen_porcentaje'] < 10)]
    if not productos_margen_bajo.empty:
        st.info(f"ℹ️ Hay {len(productos_margen_bajo)} productos con margen menor al 10%")
    
    st.markdown("---")
    
    # Top productos por margen
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Productos por Margen")
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
        st.subheader("📊 Gráfico de Márgenes por Producto")
        top_margen_chart = df_margen.groupby('nombre')['total_margen'].sum().sort_values(ascending=True).tail(10)
        fig_margen = px.bar(
            x=top_margen_chart.values,
            y=top_margen_chart.index,
            orientation='h',
            title='Top 10 Productos por Margen Total',
            labels={'x': 'Margen Total ($)', 'y': 'Producto'}
        )
        st.plotly_chart(fig_margen, use_container_width=True, key="chart_margen_productos")


# ---------------------------------------------------------------------------
# Tabla de Datos
# ---------------------------------------------------------------------------

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
    
    # Seleccionar columnas a mostrar
    default_cols = ['fecha_venta', 'nombre', 'precio', 'cantidad', 'total_venta', 
                   'vendedor', 'familia', 'metodo', 'proveedor_moda', 'precio_promedio_compra', 
                   'margen', 'margen_porcentaje']
    available_cols = [col for col in default_cols if col in df_filtered.columns]
    
    cols_selected = st.multiselect(
        "Seleccionar columnas a mostrar",
        options=available_cols,
        default=available_cols[:8],  # Mostrar primeras 8 por defecto
        key="columnas_select"
    )
    
    if cols_selected:
        df_display = df_filtered[cols_selected].copy()
    else:
        df_display = df_filtered[available_cols].copy()
    
    # Formatear datos para mostrar
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


# ---------------------------------------------------------------------------
# Exportación
# ---------------------------------------------------------------------------

def render_export(df: pd.DataFrame):
    """Renderiza la funcionalidad de exportación."""
    if df.empty:
        return
    
    st.header("💾 Exportar Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Exportar CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar CSV",
            data=csv,
            file_name=f"reporte_ventas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Exportar Excel
        try:
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Ventas')
                
                # Agregar hoja de resumen
                resumen = pd.DataFrame({
                    'Métrica': ['Total Ventas', 'Total Registros', 'Precio Promedio', 
                               'Margen Promedio', 'Margen Total'],
                    'Valor': [
                        f"${df['total_venta'].sum():,.2f}",
                        len(df),
                        f"${df['precio'].mean():,.2f}",
                        f"${df['margen'].mean():,.2f}",
                        f"${df['total_margen'].sum():,.2f}"
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


# ---------------------------------------------------------------------------
# Funciones para Sección de Pedidos - Sidebar y Métricas
# ---------------------------------------------------------------------------

def render_pedidos_sidebar(df: pd.DataFrame):
    """Renderiza el sidebar con selector de proveedor y filtros para pedidos."""
    st.sidebar.header("🛒 Gestión de Pedidos")
    
    # Selector de proveedor
    proveedores_unicos = sorted(df['moda_proveedor'].dropna().unique().tolist())
    opciones_proveedor = ["Todos los proveedores"] + proveedores_unicos
    
    proveedor_seleccionado = st.sidebar.selectbox(
        "🏭 Seleccionar Proveedor",
        options=opciones_proveedor,
        key="proveedor_seleccionado"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filtros Avanzados")
    
    if not df.empty:
        # Filtros de Producto
        st.sidebar.markdown("**📦 Producto**")
        productos_unicos = sorted(df['nombre'].dropna().unique())
        st.sidebar.multiselect(
            "Nombre de producto",
            options=productos_unicos,
            key="pedidos_productos"
        )
        
        familias_unicas = sorted(df['familia'].dropna().unique())
        st.sidebar.multiselect(
            "Familia",
            options=familias_unicas,
            key="pedidos_familias"
        )
        
        # Rango de cantidad disponible
        if df['cantidad_disponible'].notna().any():
            cantidad_disp_min = float(df['cantidad_disponible'].min())
            cantidad_disp_max = float(df['cantidad_disponible'].max())
            st.sidebar.slider(
                "Cantidad disponible",
                min_value=cantidad_disp_min,
                max_value=cantidad_disp_max,
                value=(cantidad_disp_min, cantidad_disp_max),
                key="pedidos_cantidad_disp_range"
            )
        
        # Rango de cantidad a comprar
        if df['cantidad_a_comprar'].notna().any():
            cantidad_comprar_min = float(df['cantidad_a_comprar'].min())
            cantidad_comprar_max = float(df['cantidad_a_comprar'].max())
            st.sidebar.slider(
                "Cantidad a comprar",
                min_value=cantidad_comprar_min,
                max_value=cantidad_comprar_max,
                value=(cantidad_comprar_min, cantidad_comprar_max),
                key="pedidos_cantidad_comprar_range"
            )
        
        st.sidebar.markdown("**📊 Ventas**")
        
        # Rango de ventas 7 días
        if df['ventas_7_dias'].notna().any():
            ventas_7_min = int(df['ventas_7_dias'].min())
            ventas_7_max = int(df['ventas_7_dias'].max())
            st.sidebar.slider(
                "Ventas últimos 7 días",
                min_value=ventas_7_min,
                max_value=ventas_7_max,
                value=(ventas_7_min, ventas_7_max),
                key="pedidos_ventas_7_range"
            )
        
        # Rango de ventas 30 días
        if df['ventas_30_dias'].notna().any():
            ventas_30_min = int(df['ventas_30_dias'].min())
            ventas_30_max = int(df['ventas_30_dias'].max())
            st.sidebar.slider(
                "Ventas últimos 30 días",
                min_value=ventas_30_min,
                max_value=ventas_30_max,
                value=(ventas_30_min, ventas_30_max),
                key="pedidos_ventas_30_range"
            )
        
        # Rango de ventas 90 días
        if df['ventas_90_dias'].notna().any():
            ventas_90_min = int(df['ventas_90_dias'].min())
            ventas_90_max = int(df['ventas_90_dias'].max())
            st.sidebar.slider(
                "Ventas últimos 90 días",
                min_value=ventas_90_min,
                max_value=ventas_90_max,
                value=(ventas_90_min, ventas_90_max),
                key="pedidos_ventas_90_range"
            )
        
        st.sidebar.markdown("**💰 Rentabilidad**")
        
        # Rango de margen
        if df['margen'].notna().any():
            margen_min = float(df['margen'].min())
            margen_max = float(df['margen'].max())
            st.sidebar.slider(
                "Margen",
                min_value=margen_min,
                max_value=margen_max,
                value=(margen_min, margen_max),
                key="pedidos_margen_range"
            )
        
        # Rango de utilidad
        if df['utilidad'].notna().any():
            utilidad_min = float(df['utilidad'].min())
            utilidad_max = float(df['utilidad'].max())
            st.sidebar.slider(
                "Utilidad (%)",
                min_value=utilidad_min,
                max_value=utilidad_max,
                value=(utilidad_min, utilidad_max),
                key="pedidos_utilidad_range"
            )
        
        # Rango de precio promedio compra
        if df['precio_promedio_compra'].notna().any():
            precio_compra_min = float(df['precio_promedio_compra'].min())
            precio_compra_max = float(df['precio_promedio_compra'].max())
            st.sidebar.slider(
                "Precio promedio compra",
                min_value=precio_compra_min,
                max_value=precio_compra_max,
                value=(precio_compra_min, precio_compra_max),
                key="pedidos_precio_compra_range"
            )
        
        # Rango de precio promedio venta
        if df['precio_promedio_venta'].notna().any():
            precio_venta_min = float(df['precio_promedio_venta'].min())
            precio_venta_max = float(df['precio_promedio_venta'].max())
            st.sidebar.slider(
                "Precio promedio venta",
                min_value=precio_venta_min,
                max_value=precio_venta_max,
                value=(precio_venta_min, precio_venta_max),
                key="pedidos_precio_venta_range"
            )
        
        st.sidebar.markdown("**⏰ Tiempo**")
        
        # Días desde última compra
        if 'dias_desde_ultima_compra' in df.columns and df['dias_desde_ultima_compra'].notna().any():
            dias_min = int(df['dias_desde_ultima_compra'].min())
            dias_max = int(df['dias_desde_ultima_compra'].max())
            st.sidebar.slider(
                "Días desde última compra",
                min_value=dias_min,
                max_value=dias_max,
                value=(dias_min, dias_max),
                key="pedidos_dias_compra_range"
            )
        
        # Fecha última compra
        if df['fecha_ultima_compra'].notna().any():
            fecha_compra_min = df['fecha_ultima_compra'].min().date()
            fecha_compra_max = df['fecha_ultima_compra'].max().date()
            st.sidebar.date_input(
                "Fecha última compra (desde)",
                value=fecha_compra_min,
                key="pedidos_fecha_compra_inicio"
            )
            st.sidebar.date_input(
                "Fecha última compra (hasta)",
                value=fecha_compra_max,
                key="pedidos_fecha_compra_fin"
            )
        
        st.sidebar.markdown("**🎯 Prioridad**")
        
        # Solo productos a comprar
        st.sidebar.checkbox(
            "Solo productos a comprar",
            key="pedidos_solo_comprar"
        )
        
        # Stock bajo
        st.sidebar.checkbox(
            "Solo stock bajo",
            key="pedidos_stock_bajo"
        )
        if st.session_state.get('pedidos_stock_bajo', False):
            st.sidebar.number_input(
                "Umbral de stock bajo",
                min_value=0,
                value=10,
                key="pedidos_umbral_stock"
            )
        
        # Ordenar por
        opciones_orden = [
            'cantidad_a_comprar', 'ventas_30_dias', 'utilidad', 'margen',
            'cantidad_disponible', 'precio_ultimo_compra', 'nombre'
        ]
        st.sidebar.selectbox(
            "Ordenar por",
            options=opciones_orden,
            key="pedidos_ordenar_por"
        )
        st.sidebar.checkbox(
            "Orden descendente",
            value=True,
            key="pedidos_orden_desc"
        )
    
    # Botón limpiar filtros
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Limpiar Filtros", use_container_width=True, key="limpiar_filtros_pedidos"):
        st.rerun()
    
    return proveedor_seleccionado


def render_proveedor_metrics(df: pd.DataFrame, proveedor: str, ventas_data: dict):
    """Renderiza las métricas del proveedor seleccionado."""
    if df.empty:
        return
    
    st.header(f"📊 Dashboard del Proveedor: {proveedor}")
    
    # Calcular métricas de inventario
    productos_unicos = df['nombre'].nunique()
    total_unidades_inventario = df['cantidad_disponible'].sum()
    valor_total_inventario = df['valor_inventario'].sum()
    
    # Calcular métricas de recomendaciones
    productos_a_comprar = len(df[df['cantidad_a_comprar'] > 0])
    total_unidades_comprar = df['cantidad_a_comprar'].sum()
    valor_total_pedido = df['valor_pedido'].sum()
    
    # Métricas en columnas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📦 Productos Únicos",
            f"{productos_unicos}",
            delta=None
        )
        st.metric(
            "📊 Unidades en Inventario",
            f"{total_unidades_inventario:,.0f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "💰 Valor del Inventario",
            f"${valor_total_inventario:,.2f}",
            delta=None
        )
        st.metric(
            "🛒 Productos a Comprar",
            f"{productos_a_comprar}",
            delta=None
        )
    
    with col3:
        st.metric(
            "📈 Ventas (30 días) - Unidades",
            f"{ventas_data['total_unidades']:,.0f}",
            delta=None
        )
        st.metric(
            "💵 Ventas (30 días) - Valor",
            f"${ventas_data['total_valor']:,.2f}",
            delta=None
        )
    
    with col4:
        st.metric(
            "🛍️ Productos Vendidos (30 días)",
            f"{ventas_data['productos_unicos']}",
            delta=None
        )
        st.metric(
            "📦 Unidades a Comprar",
            f"{total_unidades_comprar:,.0f}",
            delta=None
        )
    
    # Métricas adicionales
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "💵 Valor Estimado del Pedido",
            f"${valor_total_pedido:,.2f}",
            delta=None
        )
    
    with col2:
        promedio_ventas_diarias = ventas_data['total_unidades'] / 30 if ventas_data['total_unidades'] > 0 else 0
        st.metric(
            "📊 Promedio Ventas Diarias",
            f"{promedio_ventas_diarias:,.2f} unidades/día",
            delta=None
        )
    
    st.markdown("---")


# ---------------------------------------------------------------------------
# Funciones para Tabs de Pedidos
# ---------------------------------------------------------------------------

def render_pedidos_dashboard(df: pd.DataFrame):
    """Renderiza el dashboard de pedidos con gráficos."""
    if df.empty:
        st.warning("⚠️ No hay datos para mostrar con los filtros aplicados.")
        return
    
    # Top 10 productos a comprar
    st.subheader("🏆 Top 10 Productos a Comprar")
    top_comprar = df.nlargest(10, 'cantidad_a_comprar')[['nombre', 'cantidad_a_comprar', 'valor_pedido']]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_cantidad = px.bar(
            top_comprar,
            x='cantidad_a_comprar',
            y='nombre',
            orientation='h',
            title='Cantidad a Comprar',
            labels={'cantidad_a_comprar': 'Cantidad', 'nombre': 'Producto'}
        )
        st.plotly_chart(fig_cantidad, use_container_width=True, key="chart_top_cantidad_comprar")
    
    with col2:
        fig_valor = px.bar(
            top_comprar,
            x='valor_pedido',
            y='nombre',
            orientation='h',
            title='Valor Estimado por Producto',
            labels={'valor_pedido': 'Valor ($)', 'nombre': 'Producto'}
        )
        st.plotly_chart(fig_valor, use_container_width=True, key="chart_top_valor_comprar")
    
    # Stock actual vs Cantidad a comprar
    st.subheader("📊 Stock Actual vs Cantidad a Comprar")
    df_comparacion = df.nlargest(15, 'cantidad_a_comprar')[['nombre', 'cantidad_disponible', 'cantidad_a_comprar']]
    
    fig_comparacion = go.Figure()
    fig_comparacion.add_trace(go.Bar(
        x=df_comparacion['nombre'],
        y=df_comparacion['cantidad_disponible'],
        name='Stock Actual',
        marker_color='#1f77b4'
    ))
    fig_comparacion.add_trace(go.Bar(
        x=df_comparacion['nombre'],
        y=df_comparacion['cantidad_a_comprar'],
        name='Cantidad a Comprar',
        marker_color='#ff7f0e'
    ))
    fig_comparacion.update_layout(
        xaxis=dict(tickangle=45),
        barmode='group',
        height=500
    )
    st.plotly_chart(fig_comparacion, use_container_width=True, key="chart_stock_vs_comprar")
    
    # Tendencias de ventas
    st.subheader("📈 Tendencias de Ventas por Período")
    df_tendencias = df.nlargest(10, 'ventas_30_dias')
    fig_tendencias = go.Figure()
    
    fig_tendencias.add_trace(go.Bar(
        x=df_tendencias['nombre'],
        y=df_tendencias['ventas_7_dias'],
        name='7 días',
        marker_color='#d62728'
    ))
    fig_tendencias.add_trace(go.Bar(
        x=df_tendencias['nombre'],
        y=df_tendencias['ventas_15_dias'],
        name='15 días',
        marker_color='#ff7f0e'
    ))
    fig_tendencias.add_trace(go.Bar(
        x=df_tendencias['nombre'],
        y=df_tendencias['ventas_30_dias'],
        name='30 días',
        marker_color='#2ca02c'
    ))
    fig_tendencias.add_trace(go.Bar(
        x=df_tendencias['nombre'],
        y=df_tendencias['ventas_60_dias'],
        name='60 días',
        marker_color='#1f77b4'
    ))
    fig_tendencias.add_trace(go.Bar(
        x=df_tendencias['nombre'],
        y=df_tendencias['ventas_90_dias'],
        name='90 días',
        marker_color='#9467bd'
    ))
    
    fig_tendencias.update_layout(
        xaxis=dict(tickangle=45),
        barmode='group',
        height=500
    )
    st.plotly_chart(fig_tendencias, use_container_width=True, key="chart_tendencias_ventas")


def render_pedidos_rentabilidad(df: pd.DataFrame):
    """Renderiza el análisis de rentabilidad de pedidos."""
    if df.empty:
        return
    
    st.header("💵 Análisis de Rentabilidad")
    
    # Filtrar productos con datos de rentabilidad
    df_rent = df[df['utilidad'].notna()].copy()
    
    if df_rent.empty:
        st.info("ℹ️ No hay datos de rentabilidad para analizar.")
        return
    
    # Tabla de productos ordenados por utilidad
    st.subheader("📊 Productos Ordenados por Utilidad")
    top_rentables = df_rent.nlargest(20, 'utilidad')[
        ['nombre', 'utilidad', 'margen', 'precio_promedio_compra', 'precio_promedio_venta', 'cantidad_a_comprar']
    ]
    
    st.dataframe(
        top_rentables.style.format({
            'utilidad': '{:.2f}%',
            'margen': '${:,.2f}',
            'precio_promedio_compra': '${:,.2f}',
            'precio_promedio_venta': '${:,.2f}',
            'cantidad_a_comprar': '{:,.0f}'
        }),
        use_container_width=True
    )
    
    # Gráfico de dispersión
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Margen vs Utilidad")
        fig_scatter = px.scatter(
            df_rent.head(100),
            x='margen',
            y='utilidad',
            size='cantidad_a_comprar',
            color='precio_promedio_venta',
            hover_data=['nombre'],
            title='Relación entre Margen y Utilidad',
            labels={'margen': 'Margen ($)', 'utilidad': 'Utilidad (%)'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True, key="chart_margen_utilidad")
    
    with col2:
        st.subheader("🏆 Top 10 Productos Más Rentables")
        top_utilidad = df_rent.nlargest(10, 'utilidad')
        fig_bar = px.bar(
            top_utilidad,
            x='utilidad',
            y='nombre',
            orientation='h',
            title='Top 10 por Utilidad',
            labels={'utilidad': 'Utilidad (%)', 'nombre': 'Producto'}
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="chart_top_rentables")
    
    # Productos con baja rentabilidad
    productos_baja_rent = df_rent[df_rent['utilidad'] < 10]
    if not productos_baja_rent.empty:
        st.warning(f"⚠️ Hay {len(productos_baja_rent)} productos con utilidad menor al 10%")


def render_pedidos_ventas(df: pd.DataFrame):
    """Renderiza el análisis de ventas de pedidos."""
    if df.empty:
        return
    
    st.header("📊 Análisis de Ventas")
    
    # Gráfico de ventas por período
    st.subheader("📈 Ventas por Período")
    periodos = ['ventas_7_dias', 'ventas_15_dias', 'ventas_30_dias', 'ventas_60_dias', 'ventas_90_dias']
    df_top = df.nlargest(15, 'ventas_30_dias')
    
    fig_ventas = go.Figure()
    for periodo in periodos:
        fig_ventas.add_trace(go.Bar(
            x=df_top['nombre'],
            y=df_top[periodo],
            name=periodo.replace('ventas_', '').replace('_', ' ').title()
        ))
    
    fig_ventas.update_layout(
        xaxis=dict(tickangle=45),
        barmode='group',
        height=500
    )
    st.plotly_chart(fig_ventas, use_container_width=True, key="chart_ventas_periodo")
    
    # Tabla comparativa
    st.subheader("📋 Comparativa de Ventas por Período")
    df_comparativa = df.nlargest(20, 'ventas_30_dias')[
        ['nombre', 'ventas_7_dias', 'ventas_15_dias', 'ventas_30_dias', 'ventas_60_dias', 'ventas_90_dias']
    ]
    
    st.dataframe(
        df_comparativa.style.format({
            'ventas_7_dias': '{:,.0f}',
            'ventas_15_dias': '{:,.0f}',
            'ventas_30_dias': '{:,.0f}',
            'ventas_60_dias': '{:,.0f}',
            'ventas_90_dias': '{:,.0f}'
        }),
        use_container_width=True
    )
    
    # Identificar tendencias
    st.subheader("📊 Análisis de Tendencias")
    df_tendencias = df.copy()
    df_tendencias['tendencia_7_30'] = (df_tendencias['ventas_7_dias'] / (df_tendencias['ventas_30_dias'] / 30 * 7)).fillna(0)
    df_tendencias['tendencia'] = df_tendencias['tendencia_7_30'].apply(
        lambda x: 'Creciente' if x > 1.2 else 'Decreciente' if x < 0.8 else 'Estable'
    )
    
    col1, col2, col3 = st.columns(3)
    crecientes = len(df_tendencias[df_tendencias['tendencia'] == 'Creciente'])
    decrecientes = len(df_tendencias[df_tendencias['tendencia'] == 'Decreciente'])
    estables = len(df_tendencias[df_tendencias['tendencia'] == 'Estable'])
    
    with col1:
        st.metric("📈 Productos con Tendencia Creciente", crecientes)
    with col2:
        st.metric("📉 Productos con Tendencia Decreciente", decrecientes)
    with col3:
        st.metric("➡️ Productos Estables", estables)


def render_pedidos_tabla(df: pd.DataFrame):
    """Renderiza la tabla interactiva de pedidos."""
    if df.empty:
        return
    
    st.header("📋 Tabla de Pedidos")
    
    # Búsqueda
    search_term = st.text_input("🔍 Buscar por nombre", "", key="search_pedidos")
    
    if search_term:
        mask = df['nombre'].str.contains(search_term, case=False, na=False)
        df_filtered = df[mask]
    else:
        df_filtered = df.copy()
    
    # Seleccionar columnas
    default_cols = [
        'nombre', 'familia', 'cantidad_disponible', 'cantidad_a_comprar',
        'precio_ultimo_compra', 'precio_promedio_venta', 'margen', 'utilidad',
        'ventas_7_dias', 'ventas_30_dias', 'ventas_90_dias',
        'fecha_ultima_compra', 'cantidad_ultima_compra'
    ]
    available_cols = [col for col in default_cols if col in df_filtered.columns]
    
    cols_selected = st.multiselect(
        "Seleccionar columnas",
        options=available_cols,
        default=available_cols[:8],
        key="pedidos_columnas_select"
    )
    
    if cols_selected:
        df_display = df_filtered[cols_selected].copy()
    else:
        df_display = df_filtered[available_cols].copy()
    
    # Formatear fechas
    if 'fecha_ultima_compra' in df_display.columns:
        df_display['fecha_ultima_compra'] = df_display['fecha_ultima_compra'].dt.strftime('%Y-%m-%d')
    
    # Resaltar filas con cantidad_a_comprar > 0
    def highlight_comprar(row):
        if 'cantidad_a_comprar' in row.index and row['cantidad_a_comprar'] > 0:
            return ['background-color: #ffeb3b'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        df_display.style.apply(highlight_comprar, axis=1).format({
            'cantidad_disponible': '{:,.2f}',
            'cantidad_a_comprar': '{:,.2f}',
            'precio_ultimo_compra': '${:,.2f}',
            'precio_promedio_venta': '${:,.2f}',
            'margen': '${:,.2f}',
            'utilidad': '{:.2f}%',
            'ventas_7_dias': '{:,.0f}',
            'ventas_30_dias': '{:,.0f}',
            'ventas_90_dias': '{:,.0f}',
            'cantidad_ultima_compra': '{:,.2f}'
        }, na_rep='N/A'),
        use_container_width=True,
        height=600
    )
    
    st.caption(f"Mostrando {len(df_display):,} de {len(df):,} productos")


def render_pedidos_export(df: pd.DataFrame, proveedor: str):
    """Renderiza la funcionalidad de exportación de pedidos."""
    if df.empty:
        return
    
    st.header("💾 Exportar Pedido")
    
    # Filtrar solo productos a comprar
    df_comprar = df[df['cantidad_a_comprar'] > 0].copy()
    
    if df_comprar.empty:
        st.info("ℹ️ No hay productos a comprar con los filtros aplicados.")
        return
    
    # Resumen del pedido
    st.subheader(f"📋 Resumen del Pedido - {proveedor}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Productos a Comprar", len(df_comprar))
    with col2:
        st.metric("Total Unidades", f"{df_comprar['cantidad_a_comprar'].sum():,.0f}")
    with col3:
        st.metric("Valor Total", f"${df_comprar['valor_pedido'].sum():,.2f}")
    
    # Tabla de productos a comprar
    st.subheader("🛒 Productos a Comprar")
    df_export = df_comprar[[
        'nombre', 'familia', 'cantidad_a_comprar', 'precio_ultimo_compra', 
        'valor_pedido', 'cantidad_disponible', 'ventas_30_dias'
    ]].copy()
    df_export.columns = [
        'Producto', 'Familia', 'Cantidad a Comprar', 'Precio Unitario',
        'Valor Total', 'Stock Actual', 'Ventas 30 días'
    ]
    
    st.dataframe(
        df_export.style.format({
            'Cantidad a Comprar': '{:,.2f}',
            'Precio Unitario': '${:,.2f}',
            'Valor Total': '${:,.2f}',
            'Stock Actual': '{:,.2f}',
            'Ventas 30 días': '{:,.0f}'
        }),
        use_container_width=True
    )
    
    # Exportar CSV
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df_export.to_csv(index=False)
        st.download_button(
            label="📥 Descargar CSV",
            data=csv,
            file_name=f"pedido_{proveedor.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        try:
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_export.to_excel(writer, index=False, sheet_name='Pedido')
                
                # Resumen
                resumen = pd.DataFrame({
                    'Métrica': ['Proveedor', 'Productos', 'Total Unidades', 'Valor Total', 'Fecha'],
                    'Valor': [
                        proveedor,
                        len(df_comprar),
                        f"{df_comprar['cantidad_a_comprar'].sum():,.0f}",
                        f"${df_comprar['valor_pedido'].sum():,.2f}",
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ]
                })
                resumen.to_excel(writer, index=False, sheet_name='Resumen')
            
            excel_data = output.getvalue()
            st.download_button(
                label="📊 Descargar Excel",
                data=excel_data,
                file_name=f"pedido_{proveedor.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except ImportError:
            st.error("⚠️ openpyxl no está instalado. Instala con: pip install openpyxl")
    
    # Resumen en texto
    st.subheader("📝 Resumen para Proveedor")
    st.text_area(
        "Copia este texto para enviar al proveedor:",
        value=f"""Pedido para {proveedor}
Fecha: {datetime.now().strftime('%d/%m/%Y')}

Total de productos: {len(df_comprar)}
Total de unidades: {df_comprar['cantidad_a_comprar'].sum():,.0f}
Valor estimado: ${df_comprar['valor_pedido'].sum():,.2f}

Detalle:
{chr(10).join([f"- {row['Producto']}: {row['Cantidad a Comprar']:,.0f} unidades" for _, row in df_export.iterrows()])}
""",
        height=200,
        key="resumen_texto"
    )


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    """Función principal de la aplicación."""
    # Menú de navegación en el sidebar
    st.sidebar.title("📊 Sistema de Reportes")
    st.sidebar.markdown("---")
    
    # Selector de sección
    seccion = st.sidebar.radio(
        "Seleccionar sección",
        options=["📊 Reportes de Ventas", "🛒 Para Pedidos"],
        key="seccion_seleccionada"
    )
    
    st.sidebar.markdown("---")
    
    # Navegar según la sección seleccionada
    if seccion == "📊 Reportes de Ventas":
        render_seccion_ventas()
    elif seccion == "🛒 Para Pedidos":
        render_seccion_pedidos()


def render_seccion_ventas():
    """Renderiza la sección de reportes de ventas."""
    st.title("📊 Reportes de Ventas - Últimos 30 Días")
    st.markdown("""
    Aplicación interactiva para analizar ventas de los últimos 30 días con información de:
    - **Ventas**: precio, cantidad, método de pago, vendedor
    - **Productos**: familia desde tabla items
    - **Proveedores**: precio promedio de compra y proveedor más frecuente (últimas 3 compras)
    """)
    
    # Cargar datos
    with st.spinner("Cargando datos..."):
        df = load_data()
    
    if df.empty:
        st.error("❌ No hay datos en la tabla. Ejecuta primero el script generar_reporte_ventas_30dias.py")
        st.stop()
    
    # Sidebar con filtros
    render_sidebar_filters(df)
    
    # Aplicar filtros
    df_filtered = apply_filters(df)
    
    # Tabs para organizar contenido
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "💵 Márgenes", 
        "📈 Gráficos",
        "👤 Por Vendedor",
        "📋 Datos"
    ])
    
    with tab1:
        render_metrics(df_filtered)
        render_charts(df_filtered, key_prefix="tab1_")
    
    with tab2:
        render_margin_analysis(df_filtered)
    
    with tab3:
        render_charts(df_filtered, key_prefix="tab3_")
    
    with tab4:
        st.header("👤 Análisis por Vendedor")
        if not df_filtered.empty:
            vendedor_seleccionado = st.selectbox(
                "Seleccionar vendedor",
                options=['Todos'] + sorted(df_filtered['vendedor'].dropna().unique().tolist())
            )
            
            if vendedor_seleccionado != 'Todos':
                df_vendedor = df_filtered[df_filtered['vendedor'] == vendedor_seleccionado]
            else:
                df_vendedor = df_filtered.copy()
            
            if not df_vendedor.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Ventas", f"${df_vendedor['total_venta'].sum():,.2f}")
                    st.metric("Total Registros", len(df_vendedor))
                with col2:
                    st.metric("Productos Únicos", df_vendedor['nombre'].nunique())
                    st.metric("Promedio por Venta", f"${df_vendedor['total_venta'].mean():,.2f}")
                
                # Gráfico de ventas diarias del vendedor
                ventas_vendedor_dia = df_vendedor.groupby(df_vendedor['fecha_venta'].dt.date)['total_venta'].sum()
                fig_vendedor = px.line(
                    x=ventas_vendedor_dia.index,
                    y=ventas_vendedor_dia.values,
                    title=f'Ventas Diarias de {vendedor_seleccionado}',
                    labels={'x': 'Fecha', 'y': 'Total Ventas ($)'}
                )
                st.plotly_chart(fig_vendedor, use_container_width=True, key=f"chart_vendedor_{vendedor_seleccionado}")
    
    with tab5:
        render_data_table(df_filtered)
        st.markdown("---")
        render_export(df_filtered)


def render_seccion_pedidos():
    """Renderiza la sección de gestión de pedidos."""
    st.title("🛒 Gestión de Pedidos a Proveedores")
    st.markdown("""
    Aplicación para gestionar pedidos a proveedores con información de:
    - **Inventario**: productos, cantidades disponibles y valores
    - **Ventas**: análisis de ventas por períodos (7, 15, 30, 60, 90 días)
    - **Recomendaciones**: cantidad sugerida a comprar basada en algoritmo de retail
    - **Rentabilidad**: márgenes y utilidades por producto
    """)
    
    # Cargar datos
    with st.spinner("Cargando datos de pedidos..."):
        df_pedidos = load_pedidos_data()
    
    if df_pedidos.empty:
        st.error("❌ No hay datos en la tabla para_pedidos. Ejecuta primero el script generar_tabla_para_pedidos.py")
        st.stop()
    
    # Sidebar con selector de proveedor y filtros
    proveedor_seleccionado = render_pedidos_sidebar(df_pedidos)
    
    # Aplicar filtros
    df_filtered = apply_pedidos_filters(df_pedidos, proveedor_seleccionado)
    
    if df_filtered.empty:
        st.warning("⚠️ No hay productos que coincidan con los filtros aplicados.")
        st.stop()
    
    # Calcular ventas del proveedor si está seleccionado
    ventas_data = {'productos_unicos': 0, 'total_unidades': 0, 'total_valor': 0}
    if proveedor_seleccionado and proveedor_seleccionado != "Todos los proveedores":
        ventas_data = load_ventas_proveedor_30dias(proveedor_seleccionado)
    
    # Mostrar métricas del proveedor
    if proveedor_seleccionado and proveedor_seleccionado != "Todos los proveedores":
        render_proveedor_metrics(df_filtered, proveedor_seleccionado, ventas_data)
    else:
        st.info("ℹ️ Selecciona un proveedor específico para ver métricas detalladas.")
    
    # Tabs para organizar contenido
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "💵 Rentabilidad",
        "📈 Ventas",
        "📋 Tabla de Pedidos",
        "💾 Exportar"
    ])
    
    with tab1:
        render_pedidos_dashboard(df_filtered)
    
    with tab2:
        render_pedidos_rentabilidad(df_filtered)
    
    with tab3:
        render_pedidos_ventas(df_filtered)
    
    with tab4:
        render_pedidos_tabla(df_filtered)
    
    with tab5:
        if proveedor_seleccionado and proveedor_seleccionado != "Todos los proveedores":
            render_pedidos_export(df_filtered, proveedor_seleccionado)
        else:
            st.info("ℹ️ Selecciona un proveedor específico para exportar el pedido.")


if __name__ == "__main__":
    main()

