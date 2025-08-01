import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# Configuración de la página
st.set_page_config(page_title="Curva de Sistema de Bombeo", layout="wide")
st.title("Cálculo de la Curva de un Sistema de Bombeo")

# Datos de entrada en el sidebar
st.sidebar.header("Parámetros del Sistema")

# Datos del fluido
st.sidebar.subheader("Propiedades del Fluido")
densidad = st.sidebar.number_input("Densidad del agua (kg/m³)", value=998.2)
viscosidad = st.sidebar.number_input("Viscosidad dinámica (Pa·s)", value=0.001002)

# Datos de la tubería
st.sidebar.subheader("Geometría de la Tubería")
longitud = st.sidebar.number_input("Longitud total de la tubería (m)", value=100.0)
diametro = st.sidebar.number_input("Diámetro interno de la tubería (m)", value=0.1)
rugosidad = st.sidebar.number_input("Rugosidad absoluta (m)", value=0.000045)

# Altura geodésica
st.sidebar.subheader("Altura Geodésica")
altura_geodesica = st.sidebar.number_input("Diferencia de altura entre entrada y salida (m)", value=10.0)

# Pérdidas secundarias
st.sidebar.subheader("Pérdidas Secundarias (Accesorios)")
num_accesorios = st.sidebar.number_input("Número de accesorios", value=5, min_value=0)
coef_perdida = st.sidebar.number_input("Coeficiente de pérdida por accesorio (K)", value=0.5)

# Datos de caudal y presión
st.header("Datos de Caudal y Presión")
st.write("Introduce los datos medidos de caudal (Q) y presiones de entrada (P1) y salida (P2):")

# Crear un DataFrame editable
data = {
    'Q (m³/s)': [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
    'P1 (Pa)': [101325]*10,
    'P2 (Pa)': [101325, 110000, 120000, 135000, 155000, 180000, 210000, 245000, 285000, 330000]
}
df = pd.DataFrame(data)
edited_df = st.data_editor(df, num_rows="dynamic")

# Función para calcular el factor de fricción (Colebrook-White)
def factor_friccion(Re, rugosidad_relativa):
    # Aproximación de Haaland
    return (-1.8 * np.log10((6.9/Re) + (rugosidad_relativa/3.7)**1.11))**(-2)

# Función para calcular las pérdidas
def calcular_perdidas(Q, L, D, rugosidad, num_acc, coef_acc, densidad, viscosidad):
    A = np.pi * (D**2) / 4
    v = Q / A
    Re = densidad * v * D / viscosidad
    rugosidad_relativa = rugosidad / D
    
    if Re == 0:
        return 0
    
    f = factor_friccion(Re, rugosidad_relativa)
    
    # Pérdidas primarias
    hf_primarias = f * (L/D) * (v**2) / (2 * 9.81)
    
    # Pérdidas secundarias
    hf_secundarias = num_acc * coef_acc * (v**2) / (2 * 9.81)
    
    return hf_primarias + hf_secundarias

# Función para la curva del sistema
def altura_sistema(Q, altura_geodesica, L, D, rugosidad, num_acc, coef_acc, densidad, viscosidad):
    perdidas = calcular_perdidas(Q, L, D, rugosidad, num_acc, coef_acc, densidad, viscosidad)
    return altura_geodesica + perdidas

# Procesar los datos
if st.button("Calcular Curva del Sistema"):
    try:
        Q = edited_df['Q (m³/s)'].values
        P1 = edited_df['P1 (Pa)'].values
        P2 = edited_df['P2 (Pa)'].values
        
        # Convertir presiones a altura (H = (P2-P1)/(ρg) + z + pérdidas)
        g = 9.81
        H_sistema = (P2 - P1) / (densidad * g) + altura_geodesica
        
        # Ajustar una curva a los puntos
        def modelo_curva(Q, a, b, c):
            return a + b*Q + c*Q**2
        
        popt, pcov = curve_fit(modelo_curva, Q, H_sistema)
        
        # Generar puntos para la curva suavizada
        Q_smooth = np.linspace(min(Q), max(Q), 100)
        H_smooth = modelo_curva(Q_smooth, *popt)
        
        # Calcular la curva teórica del sistema
        H_teorico = [altura_sistema(q, altura_geodesica, longitud, diametro, 
                                    rugosidad, num_accesorios, coef_perdida, 
                                    densidad, viscosidad) for q in Q_smooth]
        
        # Mostrar resultados
        st.subheader("Resultados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Parámetros de la curva ajustada:")
            st.write(f"a (altura estática) = {popt[0]:.2f} m")
            st.write(f"b (término lineal) = {popt[1]:.2f} s/m²")
            st.write(f"c (término cuadrático) = {popt[2]:.2f} s²/m⁵")
            
            st.write("\nEcuación de la curva del sistema:")
            st.write(f"H = {popt[0]:.2f} + {popt[1]:.2f}·Q + {popt[2]:.2f}·Q²")
        
        with col2:
            # Crear gráfico
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(Q, H_sistema, 'o', label='Datos medidos')
            ax.plot(Q_smooth, H_smooth, '-', label='Curva ajustada')
            ax.plot(Q_smooth, H_teorico, '--', label='Curva teórica')
            
            ax.set_xlabel('Caudal (m³/s)')
            ax.set_ylabel('Altura (m)')
            ax.set_title('Curva del Sistema de Bombeo')
            ax.grid(True)
            ax.legend()
            
            st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error al procesar los datos: {str(e)}")

# Explicación de los parámetros
st.header("Explicación de los Parámetros")
st.markdown("""
**Datos necesarios para calcular la curva del sistema:**

1. **Propiedades del fluido:**
   - Densidad (kg/m³)
   - Viscosidad dinámica (Pa·s)

2. **Geometría de la tubería:**
   - Longitud total (m)
   - Diámetro interno (m)
   - Rugosidad absoluta (m)

3. **Altura geodésica:**
   - Diferencia de altura entre entrada y salida (m)

4. **Pérdidas secundarias:**
   - Número de accesorios (codos, válvulas, etc.)
   - Coeficiente de pérdida por accesorio (K)

5. **Datos operacionales:**
   - Caudal (Q) en m³/s
   - Presión de entrada (P1) en Pa
   - Presión de salida (P2) en Pa

La curva del sistema se calcula como:
H = (P2-P1)/(ρg) + z + Σ(pérdidas)
""")