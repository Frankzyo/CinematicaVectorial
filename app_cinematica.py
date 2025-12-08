import streamlit as st
import numpy as np
import plotly.graph_objects as go 
from sympy import symbols, diff, integrate, sqrt, solve, simplify, sin, cos, tan, exp, log, lambdify, latex, nsimplify, Abs, factor, sympify, Integral
from scipy.integrate import quad
import re

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Cinemática Vectorial", layout="wide", page_icon="📈")

st.title("⚡ Cinemática Vectorial")
st.markdown("""
**Proyecto de Física:** Herramienta para analizar el movimiento en 2D.
Calcula posición, velocidad, aceleración y grafica la trayectoria.
""")

# --- MIS FUNCIONES ---

def corregir_sintaxis(texto):
    """
    Esta función arregla lo que escribo en el input.
    Si pongo '4t', le agrega el por '*' para que quede '4*t'.
    """
    if not texto: return ""
    texto = texto.lower().strip()
    texto = texto.replace('^', '**') # Python usa ** para potencia
    # Expresiones regulares para agregar multiplicacion donde falta
    texto = re.sub(r'(\d)([a-z\(])', r'\1*\2', texto)     # Ej: cambia 2t a 2*t
    texto = re.sub(r'([a-z])(\()', r'\1\2', texto)       # Ej: t( a t*(
    texto = re.sub(r'(\))([a-z\d])', r'\1*\2', texto)    # Ej: )t a )*t
    texto = texto.replace(' ', '')
    return texto

def parsear_entrada_vectorial(texto_vector):
    """
    Si pego el vector completo (x, y), esto lo separa en dos partes.
    """
    texto = texto_vector.replace('(', '').replace(')', '')
    partes = texto.split(',')
    if len(partes) == 2:
        return partes[0], partes[1]
    return None, None

def limpiar_latex(expr):
    """
    Arreglos visuales para que las fórmulas se vean bien en pantalla.
    """
    tex = latex(expr)
    tex = tex.replace(r"\log", r"\ln")             # Uso ln en vez de log
    tex = tex.replace(r"0.5", r"\frac{1}{2}")      # Prefiero fracciones a decimales
    tex = tex.replace(r"asinh", r"\text{arcsinh}") 
    return tex

# --- BARRA LATERAL (DATOS) ---
st.sidebar.header("📝 Configuración")
# Uso pestañas para organizar las formas de meter datos
tab1, tab2, tab3 = st.sidebar.tabs(["Ecuaciones", "Vector r(t)", "Ejemplos"])

x_raw, y_raw = "", ""
valor_g = st.sidebar.number_input("Gravedad (g):", value=9.8, step=0.1)
t_ini = st.sidebar.number_input("T. Inicio:", value=0.0)
t_fin = st.sidebar.number_input("T. Fin:", value=2.0)

# Opcion 1: Escribir x e y por separado
with tab1:
    in_x = st.text_input("x(t) =", value="4 - 3t", key="x1")
    in_y = st.text_input("y(t) =", value="1 + 6t - g/2*t^2", key="y1")
    if st.button("Calcular Ecuaciones", type="primary"): 
        x_raw, y_raw = in_x, in_y

# Opcion 2: Pegar el vector entero
with tab2:
    in_vec = st.text_input("r(t) =", value="(4t-3, 1+6t-4.9t^2)")
    if st.button("Calcular Vector", type="primary"):
        vx, vy = parsear_entrada_vectorial(in_vec)
        if vx and vy: x_raw, y_raw = vx, vy

# Opcion 3: Ejercicios listos para probar
with tab3:
    st.caption("Ejercicios de clase:")
    if st.button("🔵 Circular (Ej 1)"):
        x_raw, y_raw = "2cos(t)", "-2sin(t)"
        t_fin = 6.28
    if st.button("🚀 Proyectil (Ej 2 PDF)"):
        x_raw, y_raw = "4 - 3t", "1 + 6t - g/2 * t^2"
    if st.button("✏️ Mi Ejemplo"):
        x_raw, y_raw = "4t - 3", "-5t^2 + 8t"

# --- AQUÍ EMPIEZAN LOS CÁLCULOS ---
if x_raw and y_raw:
    # 1. Primero limpio el texto que escribí
    x_clean = corregir_sintaxis(x_raw)
    y_clean = corregir_sintaxis(y_raw)
    st.sidebar.success(f"Procesando: x={x_clean}, y={y_clean}")

    # 2. Defino las variables para SymPy
    t = symbols('t', real=True)
    g = symbols('g', positive=True) # IMPORTANTE: g positivo para que la raíz cuadrada funcione bien
    
    # Lista de funciones que puede entender el programa
    ctx = {'t': t, 'g': g, 'sin': sin, 'cos': cos, 'sqrt': sqrt, 'exp': exp, 'ln': log, 'log': log, 'pi': np.pi}

    try:
        # 3. Intento convertir el texto a matemáticas
        try:
            # Trato de pasar los decimales a fracciones exactas
            x_sym = nsimplify(eval(x_clean, ctx), rational=True)
            y_sym = nsimplify(eval(y_clean, ctx), rational=True)
        except:
            # Si falla, uso el método normal
            x_sym = sympify(x_clean)
            y_sym = sympify(y_clean)

        # Hago una copia numérica reemplazando g por 9.8 para poder graficar
        x_num = x_sym.subs(g, valor_g)
        y_num = y_sym.subs(g, valor_g)

        # Divido la pantalla en dos columnas
        col1, col2 = st.columns([1.3, 1])

        # --- COLUMNA IZQUIERDA: RESULTADOS ---
        with col1:
            st.header("📝 Resolución Paso a Paso")
            
            # Paso 1: Posición
            st.subheader("1. Posición")
            st.latex(r"x = " + limpiar_latex(x_sym) + r", \quad y = " + limpiar_latex(y_sym))
            
            # Paso 3: Velocidad (Derivada de la posición)
            vx = diff(x_sym, t)
            vy = diff(y_sym, t)
            st.subheader("3. Velocidad")
            st.latex(r"\vec{v} = (" + limpiar_latex(vx) + ", " + limpiar_latex(vy) + ")")

            # Paso 4: Rapidez (Módulo de la velocidad)
            rapidez = simplify(sqrt(vx**2 + vy**2))
            st.subheader("4. Rapidez")
            st.latex(r"\dot{s} = " + limpiar_latex(rapidez))

            # Paso 5: Aceleración (Derivada de la velocidad)
            ax = diff(vx, t)
            ay = diff(vy, t)
            mod_a = simplify(sqrt(ax**2 + ay**2))
            st.subheader("5. Aceleración")
            st.latex(r"\vec{a} = (" + limpiar_latex(ax) + ", " + limpiar_latex(ay) + "), \quad a = " + limpiar_latex(mod_a))

            # Paso 6: Aceleración Tangencial
            st.subheader("6. Ac. Tangencial")
            prod = simplify(vx * ax + vy * ay)
            at = simplify(prod / rapidez)
            # Muestro la parte factorizada para ver de donde salen los números
            st.latex(r"a_t = \frac{" + limpiar_latex(factor(prod)) + "}{" + limpiar_latex(rapidez) + "} = " + limpiar_latex(at))

            # Paso 7: Aceleración Normal
            st.subheader("7. Ac. Normal")
            det = simplify(Abs(vx * ay - vy * ax))
            an = simplify(det / rapidez)
            st.latex(r"a_n = \frac{" + limpiar_latex(det) + "}{" + limpiar_latex(rapidez) + "} = " + limpiar_latex(an))
            
            # Paso 8: Radio de Curvatura
            try:
                rho = simplify(rapidez**2 / an)
                st.subheader("8. Radio de Curvatura")
                st.latex(r"\rho = " + limpiar_latex(rho))
            except: pass

            # Paso 9: Distancia Recorrida
            st.subheader("9. Distancia Recorrida")
            
            st.markdown("**Integral a resolver:**")
            st.latex(r"s(t) = \int " + limpiar_latex(rapidez) + " dt")
            
            # Intento resolver la integral simbólicamente
            try:
                s_primitive = integrate(rapidez, t)
                if isinstance(s_primitive, Integral):
                    st.info("La integral es muy difícil para mostrar la fórmula cerrada.")
                else:
                    try:
                        s_display = simplify(s_primitive.rewrite(log))
                    except:
                        s_display = s_primitive
                    st.markdown("**Resultado:**")
                    st.latex(r"s(t) = " + limpiar_latex(s_display))
                    if "asinh" in str(s_display):
                         st.caption("Nota: `asinh` es lo mismo que usar logaritmos naturales (ln).")
            except: pass

            # Calculo el valor numérico exacto usando scipy
            try:
                func_num = lambdify(t, rapidez.subs(g, valor_g), modules=['numpy'])
                dist, err = quad(func_num, float(t_ini), float(t_fin))
                st.success(f"📍 Distancia Total Calculada: **{dist:.4f} m**")
            except:
                st.error("No pude calcular el número exacto de la distancia.")

        # --- COLUMNA DERECHA: GRÁFICA ---
        with col2:
            st.header("📈 Gráfica Interactiva")
            try:
                # Genero los puntos para dibujar la línea
                t_vals = np.linspace(float(t_ini), float(t_fin), 400)
                fx = lambdify(t, x_num, modules=['numpy'])
                fy = lambdify(t, y_num, modules=['numpy'])
                xv = fx(t_vals)
                yv = fy(t_vals)
                
                # Arreglo por si x o y son constantes (ej: x=4)
                if np.isscalar(xv): xv = np.full_like(t_vals, xv)
                if np.isscalar(yv): yv = np.full_like(t_vals, yv)

                # Configuro la figura de Plotly
                fig = go.Figure()

                # Dibujo la trayectoria
                fig.add_trace(go.Scatter(
                    x=xv, y=yv, mode='lines', name='Trayectoria',
                    line=dict(color='#007BFF', width=4)
                ))

                # Marco el inicio y el final
                fig.add_trace(go.Scatter(
                    x=[xv[0]], y=[yv[0]], mode='markers', name='Inicio',
                    marker=dict(color='green', size=12, symbol='circle')
                ))
                fig.add_trace(go.Scatter(
                    x=[xv[-1]], y=[yv[-1]], mode='markers', name='Fin',
                    marker=dict(color='red', size=12, symbol='x')
                ))

                # Ajustes visuales de la gráfica
                fig.update_layout(
                    title="Gráfica y vs x",
                    xaxis_title="Posición X [m]",
                    yaxis_title="Posición Y [m]",
                    template="plotly_white",
                    height=600,
                    hovermode="closest",
                    xaxis=dict(scaleanchor="y", scaleratio=1), # Esto hace que no se deforme
                )

                st.plotly_chart(fig, use_container_width=True)
                st.caption("Puedes hacer zoom en la gráfica con el mouse.")

            except Exception as e:
                st.error(f"Error al graficar: {e}")

    except Exception as e:
        st.error(f"Hubo un error en el cálculo: {e}")
        st.info("Revisa si escribiste bien las ecuaciones.")

else:
    st.info("👈 Pon los datos del ejercicio a la izquierda para empezar.")