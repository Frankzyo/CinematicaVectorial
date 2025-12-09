import streamlit as st
import numpy as np
import plotly.graph_objects as go 
from sympy import symbols, diff, integrate, sqrt, solve, simplify, sin, cos, tan, exp, log, lambdify, latex, nsimplify, Abs, factor, sympify, Integral
from scipy.integrate import quad
import re

st.set_page_config(page_title="Cinemática Vectorial", layout="wide", page_icon="📈")

st.title("⚡ Cinemática Vectorial")
st.markdown("""
**Proyecto de Física:** Herramienta para analizar el movimiento en 2D.
Calcula posición, velocidad, aceleración y grafica la trayectoria.
""")

def corregir_sintaxis(texto):
    # Esta función arregla errores comunes al escribir
    # Por ejemplo, si escribo "4t", lo cambia a "4*t" para que Python entienda
    if not texto: return ""
    texto = texto.lower().strip()
    texto = texto.replace('^', '**') 
    texto = re.sub(r'(\d)([a-z\(])', r'\1*\2', texto)     
    texto = re.sub(r'([a-z])(\()', r'\1\2', texto)       
    texto = re.sub(r'(\))([a-z\d])', r'\1*\2', texto)    
    texto = texto.replace(' ', '')
    return texto

def parsear_entrada_vectorial(texto_vector):
    # Separa el vector (x, y) en dos ecuaciones individuales
    texto = texto_vector.replace('(', '').replace(')', '')
    partes = texto.split(',')
    if len(partes) == 2:
        return partes[0], partes[1]
    return None, None

def limpiar_latex(expr):
    # Arregla un poco cómo se ven las fórmulas en pantalla
    tex = latex(expr)
    tex = tex.replace(r"\log", r"\ln")             
    tex = tex.replace(r"0.5", r"\frac{1}{2}")      
    tex = tex.replace(r"asinh", r"\text{arcsinh}") 
    return tex

# --- MENÚ LATERAL ---
st.sidebar.header("Configuración")
tab1, tab2, tab3 = st.sidebar.tabs(["Ecuaciones", "Vector r(t)", "Ejemplos"])

x_raw, y_raw = "", ""
valor_g = st.sidebar.number_input("Gravedad (g):", value=9.8, step=0.1)
t_ini = st.sidebar.number_input("T. Inicio:", value=0.0)
t_fin = st.sidebar.number_input("T. Fin:", value=2.0)

# Opcion 1: Escribir x e y separado
with tab1:
    in_x = st.text_input("x(t) =", value="4 - 3t", key="x1")
    in_y = st.text_input("y(t) =", value="1 + 6t - g/2*t^2", key="y1")
    if st.button("Calcular Ecuaciones", type="primary"): 
        x_raw, y_raw = in_x, in_y

# Opcion 2: Pegar el vector completo
with tab2:
    in_vec = st.text_input("r(t) =", value="(4t-3, 1+6t-4.9t^2)")
    if st.button("Calcular Vector", type="primary"):
        vx, vy = parsear_entrada_vectorial(in_vec)
        if vx and vy: x_raw, y_raw = vx, vy

# Opcion 3: Ejercicios de ejemplo
with tab3:
    st.caption("Ejercicios:")
    if st.button("🔵 Circulo"):
        x_raw, y_raw = "2cos(t)", "-2sin(t)"
        t_fin = 6.28
    if st.button("🚀 Proyectil 1"):
        x_raw, y_raw = "4 - 3t", "1 + 6t - g/2 * t^2"
    if st.button("🚀 Proyectil 2"):
        x_raw, y_raw = "4t - 3", "-5t^2 + 8t"

# --- CÁLCULOS MATEMÁTICOS ---
if x_raw and y_raw:
    # Primero limpio el texto
    x_clean = corregir_sintaxis(x_raw)
    y_clean = corregir_sintaxis(y_raw)
    st.sidebar.success(f"Procesando: x={x_clean}, y={y_clean}")

    # Defino las variables para que SymPy pueda derivar e integrar
    t = symbols('t', real=True)
    g = symbols('g', positive=True) # g positivo ayuda a simplificar raíces
    
    ctx = {'t': t, 'g': g, 'sin': sin, 'cos': cos, 'sqrt': sqrt, 'exp': exp, 'ln': log, 'log': log, 'pi': np.pi}

    try:
        # Intento convertir el texto a una expresión matemática real
        try:
            x_sym = nsimplify(eval(x_clean, ctx), rational=True)
            y_sym = nsimplify(eval(y_clean, ctx), rational=True)
        except:
            x_sym = sympify(x_clean)
            y_sym = sympify(y_clean)

        # Creo una versión numérica reemplazando g por 9.8 para la gráfica
        x_num = x_sym.subs(g, valor_g)
        y_num = y_sym.subs(g, valor_g)

        col1, col2 = st.columns([1.3, 1])

        # --- COLUMNA 1: RESULTADOS ---
        with col1:
            st.header("Resolución Paso a Paso")
            
            # 1. Posición
            st.subheader("1. Posición")
            st.latex(r"x = " + limpiar_latex(x_sym) + r" \, [m], \quad y = " + limpiar_latex(y_sym) + r" \, [m]")
            
            # 3. Velocidad (Derivada de la posición)
            vx = diff(x_sym, t)
            vy = diff(y_sym, t)
            st.subheader("3. Velocidad")
            st.latex(r"\vec{v} = (" + limpiar_latex(vx) + ", " + limpiar_latex(vy) + r") \, [m/s]")

            # 4. Rapidez (Módulo)
            rapidez = simplify(sqrt(vx**2 + vy**2))
            st.subheader("4. Rapidez")
            st.latex(r"\dot{s} = " + limpiar_latex(rapidez) + r" \, [m/s]")

            # 5. Aceleración (Derivada de velocidad)
            ax = diff(vx, t)
            ay = diff(vy, t)
            mod_a = simplify(sqrt(ax**2 + ay**2))
            st.subheader("5. Aceleración")
            st.latex(r"\vec{a} = (" + limpiar_latex(ax) + ", " + limpiar_latex(ay) + r") \, [m/s^2]")
            st.latex(r"a = " + limpiar_latex(mod_a) + r" \, [m/s^2]")

            # 6. Aceleración Tangencial
            st.subheader("6. Ac. Tangencial")
            prod = simplify(vx * ax + vy * ay)
            at = simplify(prod / rapidez)
            # Muestro la factorización para entender los números
            st.latex(r"a_t = \frac{" + limpiar_latex(factor(prod)) + "}{" + limpiar_latex(rapidez) + "} = " + limpiar_latex(at) + r" \, [m/s^2]")

            # 7. Aceleración Normal
            st.subheader("7. Ac. Normal")
            det = simplify(Abs(vx * ay - vy * ax))
            an = simplify(det / rapidez)
            st.latex(r"a_n = \frac{" + limpiar_latex(det) + "}{" + limpiar_latex(rapidez) + "} = " + limpiar_latex(an) + r" \, [m/s^2]")
            
            # 8. Radio de Curvatura
            try:
                rho = simplify(rapidez**2 / an)
                st.subheader("8. Radio de Curvatura")
                st.latex(r"\rho = " + limpiar_latex(rho) + r" \, [m]")
            except: pass

            # --- PASO 8.1: RADIO DE CURVATURA MÍNIMO ---
            if rho is not None: # Solo si existe un radio válido
                st.subheader("8.1. Radio de Curvatura Mínimo")
                
                # 1. Derivamos rho respecto a t
                d_rho = simplify(diff(rho, t))
                st.latex(r"\frac{d\rho}{dt} = " + limpiar_latex(d_rho) + " = 0")
                
                try:
                    # 2. Resolvemos la ecuación para t
                    # solve devuelve una lista de soluciones (tiempos)
                    tiempos_criticos = solve(d_rho, t)
                    
                    if tiempos_criticos:
                        # Tomamos la primera solución real encontrada
                        t_min = tiempos_criticos[0]
                        
                        st.markdown("**Tiempo crítico encontrado:**")
                        st.latex(r"t = " + limpiar_latex(t_min))
                        
                        valor_t_num = float(t_min.subs(g, valor_g))
                        
                        # Verificamos si el tiempo está dentro de nuestro rango de estudio
                        if float(t_ini) <= valor_t_num <= float(t_fin):
                            radio_minimo = rho.subs({t: t_min, g: valor_g}).evalf()
                            
                            st.success(f"📍 El Radio Mínimo ocurre en **t ≈ {valor_t_num:.4f} s**")
                            st.info(f"📏 Valor del Radio Mínimo: **{radio_minimo:.4f} m**")
                        else:
                            st.warning(f"Se encontró un mínimo matemático en t={valor_t_num:.2f}s, pero está fuera de tu intervalo de tiempo ({t_ini} a {t_fin}).")
                    
                    else:
                        st.info("No se encontraron puntos críticos (el radio podría ser constante, como en un círculo).")
                        
                except Exception as e:
                    st.write("No se pudo despejar t algebraicamente (la ecuación es muy compleja).")

            # 9. Distancia Recorrida
            st.subheader("9. Distancia Recorrida")
            
            st.markdown("**Integral a resolver:**")
            st.latex(r"s(t) = \int " + limpiar_latex(rapidez) + r" \, dt")
            
            # Intento resolver la integral simbólicamente
            try:
                s_primitive = integrate(rapidez, t)
                if isinstance(s_primitive, Integral):
                    st.info("La integral es muy compleja para mostrarla cerrada.")
                else:
                    try:
                        s_display = simplify(s_primitive.rewrite(log))
                    except:
                        s_display = s_primitive
                    st.markdown("**Resultado:**")
                    st.latex(r"s(t) = " + limpiar_latex(s_display) + r" \, [m]")
                    if "asinh" in str(s_display):
                        st.caption("Nota: `asinh` es lo mismo que usar logaritmos naturales (ln).")
            except: pass

            # Cálculo numérico exacto (Integración definida)
            try:
                func_num = lambdify(t, rapidez.subs(g, valor_g), modules=['numpy'])
                dist, err = quad(func_num, float(t_ini), float(t_fin))
                st.success(f"📍 Distancia Total Calculada: **{dist:.4f} m**")
            except:
                st.error("No pude calcular el número exacto de la distancia.")

        # --- COLUMNA 2: GRÁFICA INTERACTIVA ---
        with col2:
            st.header("Gráfica Interactiva")
            st.subheader("2. Trayectoria")
            try:
                # Genero 400 puntos para que la curva se vea suave
                t_vals = np.linspace(float(t_ini), float(t_fin), 400)
                
                # Convierto las funciones matemáticas a funciones de Python
                fx = lambdify(t, x_num, modules=['numpy'])
                fy = lambdify(t, y_num, modules=['numpy'])
                
                xv = fx(t_vals)
                yv = fy(t_vals)
                
                # Si x o y son constantes (ej: x=4), relleno el array
                if np.isscalar(xv): xv = np.full_like(t_vals, xv)
                if np.isscalar(yv): yv = np.full_like(t_vals, yv)

                # Creo la figura con Plotly
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

                # Configuración visual
                fig.update_layout(
                    title="Gráfica y vs x",
                    xaxis_title="Posición X [m]",
                    yaxis_title="Posición Y [m]",
                    template="plotly_white",
                    height=600,
                    hovermode="closest",
                    xaxis=dict(scaleanchor="y", scaleratio=1),
                )

                st.plotly_chart(fig, use_container_width=True)
                st.caption("Puedes hacer zoom en la gráfica con el mouse.")

            except Exception as e:
                st.error(f"Error al graficar: {e}")

    except Exception as e:
        st.error(f"Hubo un error en el cálculo: {e}")
        st.info("Revisa si escribiste bien las ecuaciones.")

else:
    st.info("Pon los datos del ejercicio a la izquierda para empezar.")