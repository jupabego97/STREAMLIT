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
    page_title="Reportes de Ventas - 30 Días",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
DB_URL_ENV = "DATABASE_URL"
TABLE_NAME = "reportes_ventas_30dias"


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
        query = f"SELECT * FROM {TABLE_NAME} ORDER BY fecha_venta DESC, nombre"
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
# Main App
# ---------------------------------------------------------------------------

def main():
    """Función principal de la aplicación."""
    # Título y descripción
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


if __name__ == "__main__":
    main()

