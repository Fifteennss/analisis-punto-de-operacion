import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# Configuración de la página
st.set_page_config(page_title="Curva de Sistema de Bombeo", layout="wide")
st.title("Cálculo de la Curva de un Sistema de Bombeo")

# Funciones de conversión de unidades
def convertir_presion(valor, unidad_in, unidad_out='Pa'):
    factores = {
        'Pa': 1,
        'kPa': 1000,
        'bar': 100000,
        'psi': 6894.76,
        'inHg': 3386.39,
        'mmHg': 133.322
    }
    return valor * factores[unidad_in] / factores[unidad_out]

def convertir_longitud(valor, unidad_in, unidad_out='m'):
    factores = {
        'm': 1,
        'cm': 0.01,
        'mm': 0.001,
        'ft': 0.3048,
        'in': 0.0254
    }
    return valor * factores[unidad_in] / factores[unidad_out]

def convertir_caudal(valor, unidad_in, unidad_out='m³/s'):
    factores = {
        'm³/s': 1,
        'l/s': 0.001,
        'm³/h': 1/3600,
        'gal/min': 6.30902e-5
    }
    return valor * factores[unidad_in] / factores[unidad_out]

def convertir_densidad(valor, unidad_in, unidad_out='kg/m³'):
    factores = {
        'kg/m³': 1,
        'g/cm³': 1000,
        'lb/ft³': 16.0185
    }
    return valor * factores[unidad_in] / factores[unidad_out]

def convertir_viscosidad(valor, unidad_in, unidad_out='Pa·s'):
    factores = {
        'Pa·s': 1,
        'cP': 0.001,
        'lb/(ft·s)': 1.48816
    }
    return valor * factores[unidad_in] / factores[unidad_out]

# Datos de entrada en el sidebar
st.sidebar.header("Parámetros del Sistema")

# Datos del fluido
st.sidebar.subheader("Propiedades del Fluido")
col1, col2 = st.sidebar.columns(2)
with col1:
    densidad_val = st.number_input("Densidad del agua", value=998.2)
    densidad_unidad = st.selectbox("Unidad densidad", ['kg/m³', 'g/cm³', 'lb/ft³'])
with col2:
    viscosidad_val = st.number_input("Viscosidad dinámica", value=0.001002,format="%.3f")
    viscosidad_unidad = st.selectbox("Unidad viscosidad", ['Pa·s', 'cP', 'lb/(ft·s)'])

# Convertir unidades del fluido
densidad = convertir_densidad(densidad_val, densidad_unidad)
viscosidad = convertir_viscosidad(viscosidad_val, viscosidad_unidad)

# Datos de la tubería
st.sidebar.subheader("Geometría de la Tubería")
col1, col2 = st.sidebar.columns(2)
with col1:
    longitud_val = st.number_input("Longitud total de la tubería", value=100.0)
    longitud_unidad = st.selectbox("Unidad longitud", ['m', 'ft', 'in', 'cm', 'mm'])
with col2:
    diametro_val = st.number_input("Diámetro interno de la tubería", value=0.1)
    diametro_unidad = st.selectbox("Unidad diámetro", ['m', 'ft', 'in', 'cm', 'mm'])

col1, col2 = st.sidebar.columns(2)
with col1:
    rugosidad_val = st.number_input("Rugosidad absoluta", value=0.045)
with col2:
    rugosidad_unidad = st.selectbox("Unidad rugosidad", ['mm', 'm', 'ft', 'in'])

# Convertir unidades de tubería
longitud = convertir_longitud(longitud_val, longitud_unidad)
diametro = convertir_longitud(diametro_val, diametro_unidad)
rugosidad = convertir_longitud(rugosidad_val, rugosidad_unidad)

# Altura geodésica
st.sidebar.subheader("Altura Geodésica")
col1, col2 = st.sidebar.columns(2)
with col1:
    altura_geodesica_val = st.number_input("Diferencia de altura", value=10.0)
with col2:
    altura_unidad = st.selectbox("Unidad altura", ['m', 'ft', 'in'])
altura_geodesica = convertir_longitud(altura_geodesica_val, altura_unidad)

# Pérdidas secundarias
st.sidebar.subheader("Pérdidas Secundarias (Accesorios)")
num_accesorios = st.sidebar.number_input("Número de accesorios", value=5, min_value=0)
coef_perdida = st.sidebar.number_input("Coeficiente de pérdida por accesorio (K)", value=0.5)

# Datos de caudal y presión
st.header("Datos de Caudal y Presión")
st.write("Introduce los datos medidos:")

# Selección de unidades para los datos
col1, col2, col3 = st.columns(3)
with col1:
    caudal_unidad = st.selectbox("Unidad caudal", ['m³/s', 'l/s', 'm³/h', 'gal/min'])
with col2:
    presion_unidad_in = st.selectbox("Unidad presión entrada", ['Pa', 'kPa', 'bar', 'psi', 'inHg', 'mmHg'])
with col3:
    presion_unidad_out = st.selectbox("Unidad presión salida", ['Pa', 'kPa', 'bar', 'psi', 'inHg', 'mmHg'])

# Crear un DataFrame editable
data = {
    f'Q ({caudal_unidad})': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    f'P1 ({presion_unidad_in})': [0]*10 if presion_unidad_in != 'inHg' else [29.92]*10,
    f'P2 ({presion_unidad_out})': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] if presion_unidad_out != 'inHg' else [29.92, 30, 31, 32, 33, 34, 35, 36, 37, 38]
}
df = pd.DataFrame(data)
edited_df = st.data_editor(df, num_rows="dynamic")

# Función para calcular el factor de fricción (Colebrook-White)
def factor_friccion(Re, rugosidad_relativa):
    # Aproximación de Haaland
    if Re == 0:
        return 0
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
        # Obtener datos del dataframe
        Q_col = [col for col in edited_df.columns if 'Q (' in col][0]
        P1_col = [col for col in edited_df.columns if 'P1 (' in col][0]
        P2_col = [col for col in edited_df.columns if 'P2 (' in col][0]
        
        Q_data = edited_df[Q_col].values
        P1_data = edited_df[P1_col].values
        P2_data = edited_df[P2_col].values
        
        # Convertir unidades de los datos
        Q = np.array([convertir_caudal(q, caudal_unidad) for q in Q_data])
        P1 = np.array([convertir_presion(p, presion_unidad_in) for p in P1_data])
        P2 = np.array([convertir_presion(p, presion_unidad_out) for p in P2_data])
        
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
            
            st.write("\nParámetros convertidos a unidades base:")
            st.write(f"Densidad: {densidad:.2f} kg/m³")
            st.write(f"Viscosidad: {viscosidad:.6f} Pa·s")
            st.write(f"Longitud tubería: {longitud:.2f} m")
            st.write(f"Diámetro tubería: {diametro:.4f} m")
            st.write(f"Rugosidad: {rugosidad:.6f} m")
            st.write(f"Altura geodésica: {altura_geodesica:.2f} m")
        
        with col2:
            # Crear gráfico
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(Q, H_sistema, 'o', label='Datos medidos')
            ax.plot(Q_smooth, H_smooth, '-', label='Curva ajustada')
            ax.plot(Q_smooth, H_teorico, '--', label='Curva teórica')
            
            ax.set_xlabel(f'Caudal ({caudal_unidad})')
            ax.set_ylabel('Altura (m)')
            ax.set_title('Curva del Sistema de Bombeo')
            ax.grid(True)
            ax.legend()
            
            st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error al procesar los datos: {str(e)}")

# Explicación de las unidades
st.header("Información sobre Unidades")
st.markdown("""
**Conversiones de unidades disponibles:**

1. **Presión:**
   - Pascal (Pa)
   - Kilopascal (kPa)
   - Bar (bar)
   - Libras por pulgada cuadrada (psi)
   - Pulgadas de mercurio (inHg)
   - Milímetros de mercurio (mmHg)

2. **Longitud:**
   - Metros (m)
   - Centímetros (cm)
   - Milímetros (mm)
   - Pies (ft)
   - Pulgadas (in)

3. **Caudal:**
   - Metros cúbicos por segundo (m³/s)
   - Litros por segundo (l/s)
   - Metros cúbicos por hora (m³/h)
   - Galones por minuto (gal/min)

4. **Densidad:**
   - Kilogramos por metro cúbico (kg/m³)
   - Gramos por centímetro cúbico (g/cm³)
   - Libras por pie cúbico (lb/ft³)

5. **Viscosidad:**
   - Pascal-segundo (Pa·s)
   - Centipoise (cP)
   - Libra por pie-segundo (lb/(ft·s))
""")