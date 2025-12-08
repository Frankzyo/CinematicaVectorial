import streamlit as st
import numpy as np
import plotly.graph_objects as go  # <--- IMPORTANTE: Librería para gráficos interactivos
from sympy import symbols, diff, integrate, sqrt, solve, simplify, sin, cos, tan, exp, log, lambdify, latex, nsimplify, Abs, factor, sympify, Integral
from scipy.integrate import quad
import re

# --- CONFIGURACIÓN DEL PROYECTO ---
st.set_page_config(page_title="Proyecto Cinemática", layout="wide", page_icon="📈")

st.title("⚡ Cinemática Vectorial")
st.markdown("""
**Herramienta de Análisis:** Calcula vectores cinemáticos, radio de curvatura y distancia recorrida.
""")

def corregir_sintaxis(texto):
    """Corrige errores comunes de escritura 4t -> 4*t"""
    if not texto: return ""
    texto = texto.lower().strip()
    texto = texto.replace('^', '**')
    texto = re.sub(r'(\d)([a-z\(])', r'\1*\2', texto)
    texto = re.sub(r'([a-z])(\()', r'\1\2', texto)
    texto = re.sub(r'(\))([a-z\d])', r'\1*\2', texto)
    texto = texto.replace(' ', '')
    return texto

def parsear_entrada_vectorial(texto_vector):
    """Separa un vector (x, y) en dos ecuaciones"""
    texto = texto_vector.replace('(', '').replace(')', '')
    partes = texto.split(',')
    if len(partes) == 2:
        return partes[0], partes[1]
    return None, None

def limpiar_latex(expr):
    """Mejora la visualización de las fórmulas"""
    tex = latex(expr)
    tex = tex.replace(r"\log", r"\ln")
    tex = tex.replace(r"0.5", r"\frac{1}{2}")
    tex = tex.replace(r"asinh", r"\text{arcsinh}")
    return tex

# --- BARRA LATERAL (CONFIGURACIÓN) ---
st.sidebar.header("📝 Datos de Entrada")
tab1, tab2, tab3 = st.sidebar.tabs(["Ecuaciones", "Vector r(t)", "Ejemplos"])

x_raw, y_raw = "", ""
valor_g = st.sidebar.number_input("Gravedad (g):", value=9.8, step=0.1)
t_ini = st.sidebar.number_input("T. Inicio:", value=0.0)
t_fin = st.sidebar.number_input("T. Fin:", value=2.0)

# Pestañas de entrada
with tab1:
    in_x = st.text_input("x(t) =", value="4 - 3t", key="x1")
    in_y = st.text_input("y(t) =", value="1 + 6t - g/2t^2", key="y1")
    if st.button("Calcular", type="primary"): x_raw, y_raw = in_x, in_y

with tab2:
    in_vec = st.text_input("r(t) =", value="(4t-3, 1+6t-4.9t^2)")
    if st.button("Calcular Vector", type="primary"):
        vx, vy = parsear_entrada_vectorial(in_vec)
        if vx and vy: x_raw, y_raw = vx, vy

with tab3:
    if st.button("🔵 Circular (Ej 1)"):
        x_raw, y_raw = "2cos(t)", "-2sin(t)"
        t_fin = 6.28
    if st.button("🚀 Proyectil (Ej 2 PDF)"):
        x_raw, y_raw = "4 - 3t", "1 + 6t - g/2 * t^2"
    if st.button("✏️ Tu Ejemplo Manual"):
        x_raw, y_raw = "4t - 3", "-5t^2 + 8t"

# --- PROCESAMIENTO ---
if x_raw and y_raw:
    # 1. Corrección de sintaxis
    x_clean = corregir_sintaxis(x_raw)
    y_clean = corregir_sintaxis(y_raw)
    st.sidebar.success(f"Procesando: x={x_clean}, y={y_clean}")

    # 2. Definición de Símbolos
    t = symbols('t', real=True)
    g = symbols('g', positive=True) # Clave para simplificar raíces
    ctx = {'t': t, 'g': g, 'sin': sin, 'cos': cos, 'sqrt': sqrt, 'exp': exp, 'ln': log, 'log': log, 'pi': np.pi}

    try:
        # 3. Interpretación Matemática
        try:
            x_sym = nsimplify(eval(x_clean, ctx), rational=True)
            y_sym = nsimplify(eval(y_clean, ctx), rational=True)
        except:
            x_sym = sympify(x_clean)
            y_sym = sympify(y_clean)

        # Variables numéricas (para la gráfica)
        x_num = x_sym.subs(g, valor_g)
        y_num = y_sym.subs(g, valor_g)

        # Dividimos la pantalla: Resultados a la izquierda, Gráfica a la derecha
        col1, col2 = st.columns([1.3, 1])

        # --- COLUMNA 1: MATEMÁTICAS ---
        with col1:
            st.header("📝 Solución Paso a Paso")
            
            # Pasos 1 al 5 (Cinemática Básica)
            st.subheader("1. Posición")
            st.latex(r"x = " + limpiar_latex(x_sym) + r", \quad y = " + limpiar_latex(y_sym))
            
            vx = diff(x_sym, t)
            vy = diff(y_sym, t)
            st.subheader("3. Velocidad")
            st.latex(r"\vec{v} = (" + limpiar_latex(vx) + ", " + limpiar_latex(vy) + ")")

            rapidez = simplify(sqrt(vx**2 + vy**2))
            st.subheader("4. Rapidez")
            st.latex(r"\dot{s} = " + limpiar_latex(rapidez))

            ax = diff(vx, t)
            ay = diff(vy, t)
            mod_a = simplify(sqrt(ax**2 + ay**2))
            st.subheader("5. Aceleración")
            st.latex(r"\vec{a} = (" + limpiar_latex(ax) + ", " + limpiar_latex(ay) + "), \quad a = " + limpiar_latex(mod_a))

            # Pasos 6 y 7 (Aceleraciones Intrínsecas)
            st.subheader("6. Ac. Tangencial")
            prod = simplify(vx * ax + vy * ay)
            at = simplify(prod / rapidez)
            st.latex(r"a_t = \frac{" + limpiar_latex(factor(prod)) + "}{" + limpiar_latex(rapidez) + "} = " + limpiar_latex(at))

            st.subheader("7. Ac. Normal")
            det = simplify(Abs(vx * ay - vy * ax))
            an = simplify(det / rapidez)
            st.latex(r"a_n = \frac{" + limpiar_latex(det) + "}{" + limpiar_latex(rapidez) + "} = " + limpiar_latex(an))
            
            try:
                rho = simplify(rapidez**2 / an)
                st.subheader("8. Radio de Curvatura")
                st.latex(r"\rho = " + limpiar_latex(rho))
            except: pass

            # Paso 9 (Integral de Distancia)
            st.subheader("9. Distancia Recorrida")
            
            st.markdown("**Planteamiento Integral:**")
            st.latex(r"s(t) = \int " + limpiar_latex(rapidez) + " dt")
            
            # Resolución Simbólica
            try:
                s_primitive = integrate(rapidez, t)
                if isinstance(s_primitive, Integral):
                    st.info("La integral simbólica es muy compleja.")
                else:
                    # Intento mostrar logs en vez de asinh
                    try:
                        s_display = simplify(s_primitive.rewrite(log))
                    except:
                        s_display = s_primitive
                    st.markdown("**Función Primitiva:**")
                    st.latex(r"s(t) = " + limpiar_latex(s_display))
            except: pass

            # Resultado Numérico
            try:
                func_num = lambdify(t, rapidez.subs(g, valor_g), modules=['numpy'])
                dist, err = quad(func_num, float(t_ini), float(t_fin))
                st.success(f"📍 Distancia Total (t={t_ini} a {t_fin}): **{dist:.4f} m**")
            except:
                st.error("No se pudo calcular la distancia numérica.")

        # --- COLUMNA 2: GRÁFICA INTERACTIVA (PLOTLY) ---
        with col2:
            st.header("📈 Trayectoria Interactiva")
            try:
                # 1. Generar datos numéricos
                t_vals = np.linspace(float(t_ini), float(t_fin), 400)
                fx = lambdify(t, x_num, modules=['numpy'])
                fy = lambdify(t, y_num, modules=['numpy'])
                xv = fx(t_vals)
                yv = fy(t_vals)
                
                # Manejo de constantes (si x=4, convertir a array de 4s)
                if np.isscalar(xv): xv = np.full_like(t_vals, xv)
                if np.isscalar(yv): yv = np.full_like(t_vals, yv)

                # 2. Crear la Figura Interactiva
                fig = go.Figure()

                # Trazo de la curva
                fig.add_trace(go.Scatter(
                    x=xv, y=yv, 
                    mode='lines', 
                    name='Trayectoria',
                    line=dict(color='#007BFF', width=4)
                ))

                # Puntos de Inicio y Fin
                fig.add_trace(go.Scatter(
                    x=[xv[0]], y=[yv[0]], 
                    mode='markers', name='Inicio',
                    marker=dict(color='green', size=12)
                ))
                fig.add_trace(go.Scatter(
                    x=[xv[-1]], y=[yv[-1]], 
                    mode='markers', name='Fin',
                    marker=dict(color='red', size=12)
                ))

                # 3. Configuración del diseño (Layout)
                fig.update_layout(
                    title="Gráfica y vs x",
                    xaxis_title="Posición X [m]",
                    yaxis_title="Posición Y [m]",
                    template="plotly_white", # Fondo blanco limpio
                    height=600,
                    hovermode="closest",
                    # Esto es vital: Ejes a escala 1:1 para no deformar círculos
                    xaxis=dict(scaleanchor="y", scaleratio=1),
                )

                # Mostrar la gráfica
                st.plotly_chart(fig, use_container_width=True)
                st.caption("🔍 Usa el mouse para hacer zoom, moverte y ver coordenadas.")

            except Exception as e:
                st.error(f"Error al graficar: {e}")

    except Exception as e:
        st.error(f"Error en el cálculo: {e}")
        st.info("Revisa la sintaxis. Asegúrate de cerrar paréntesis.")

else:
    st.info("👈 Ingresa los datos del ejercicio en el menú lateral.")