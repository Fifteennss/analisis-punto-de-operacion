import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Configuración de la página
st.set_page_config(page_title="Curva de Sistema de Bombeo", layout="wide")
st.title("Cálculo de la Curva de un Sistema de Bombeo")

# Funciones de conversión de unidades mejoradas
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
        'l/min': 0.001/60,
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
    factores_in = {
        'Pa·s': 1,
        'cP': 0.001,
        'lb/(ft·s)': 1.48816
    }
    factores_out = {
        'Pa·s': 1,
        'cP': 1000,
        'lb/(ft·s)': 0.67197
    }
    return valor * factores_in[unidad_in] / factores_out[unidad_out]

# Función mejorada para el factor de fricción
def factor_friccion(Re, rugosidad_relativa):
    if Re < 2000:
        return 64 / Re if Re > 0 else 0
    elif 2000 <= Re < 4000:
        f_lam = 64 / 2000
        f_turb = (-1.8 * np.log10((6.9/4000) + (rugosidad_relativa/3.7)**1.11))**(-2)
        return f_lam + (f_turb - f_lam) * ((Re - 2000) / 2000)
    else:
        return (-1.8 * np.log10((6.9/Re) + (rugosidad_relativa/3.7)**1.11))**(-2)

# Función mejorada para calcular las pérdidas
def calcular_perdidas(Q, L, D, rugosidad, coef_perdida_total, densidad, viscosidad):
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
    hf_secundarias = coef_perdida_total * (v**2) / (2 * 9.81)
    
    return hf_primarias + hf_secundarias

# Función para la curva del sistema
def altura_sistema(Q, altura_geodesica, L, D, rugosidad, coef_perdida_total, densidad, viscosidad):
    if Q == 0:
        return altura_geodesica
    perdidas = calcular_perdidas(Q, L, D, rugosidad, coef_perdida_total, densidad, viscosidad)
    return altura_geodesica + perdidas

# Interfaz de usuario
st.sidebar.header("Parámetros del Sistema")

# Datos del fluido
st.sidebar.subheader("Propiedades del Fluido")
col1, col2 = st.sidebar.columns(2)
with col1:
    densidad_val = st.number_input("Densidad del agua", value=998.2, format="%.2f")
    densidad_unidad = st.selectbox("Unidad densidad", ['kg/m³', 'g/cm³', 'lb/ft³'])
with col2:
    viscosidad_val = st.number_input("Viscosidad dinámica", value=1.002, format="%.4f")
    viscosidad_unidad = st.selectbox("Unidad viscosidad", ['cP', 'Pa·s', 'lb/(ft·s)'])

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
    rugosidad_val = st.number_input("Rugosidad absoluta", value=0.0015, format="%.4f")
with col2:
    rugosidad_unidad = st.selectbox("Unidad rugosidad", ['mm', 'm', 'ft', 'in'])

# Altura geodésica
st.sidebar.subheader("Altura Geodésica")
col1, col2 = st.sidebar.columns(2)
with col1:
    altura_geodesica_val = st.number_input("Diferencia de altura", value=10.0)
with col2:
    altura_unidad = st.selectbox("Unidad altura", ['m', 'ft', 'in'])

# Pérdidas secundarias mejoradas
st.sidebar.subheader("Pérdidas Secundarias (Accesorios)")
num_tipos = st.sidebar.number_input("Número de tipos de accesorios", 1, 10, 1)
coef_perdida_total = 0.0

for i in range(num_tipos):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        num = st.number_input(f"Número accesorios tipo {i+1}", 0, 100, 1)
    with col2:
        k = st.number_input(f"Coeficiente K tipo {i+1}", 0.0, 15.0, 0.5)
    coef_perdida_total += num * k

# Datos de caudal y presión
st.header("Datos de Caudal y Presión")
st.write("Introduce los datos medidos:")

col1, col2, col3 = st.columns(3)
with col1:
    caudal_unidad = st.selectbox("Unidad caudal", ['m³/s', 'l/min', 'm³/h', 'gal/min'])
with col2:
    presion_unidad_in = st.selectbox("Unidad presión entrada", ['Pa', 'kPa', 'bar', 'psi', 'inHg', 'mmHg'])
with col3:
    presion_unidad_out = st.selectbox("Unidad presión salida", ['Pa', 'kPa', 'bar', 'psi', 'inHg', 'mmHg'])

# Crear un DataFrame editable
data = {
    f'Q ({caudal_unidad})': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    f'P1 ({presion_unidad_in})': [0]*10 if presion_unidad_in != 'inHg' else [29.92]*10,
    f'P2 ({presion_unidad_out})': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] if presion_unidad_out != 'inHg' else [29.92, 30, 31, 32, 33, 34, 35, 36, 37, 38]
}
df = pd.DataFrame(data)
edited_df = st.data_editor(df, num_rows="dynamic")

# Procesar los datos con validación
if st.button("Calcular Curva del Sistema"):
    try:
        # Validación básica
        if any(q < 0 for q in edited_df.iloc[:, 0]):
            st.error("Error: Los caudales no pueden ser negativos")
            st.stop()
            
        if diametro_val <= 0:
            st.error("Error: El diámetro debe ser positivo")
            st.stop()
            
        # Conversión de unidades
        densidad = convertir_densidad(densidad_val, densidad_unidad)
        viscosidad = convertir_viscosidad(viscosidad_val, viscosidad_unidad)
        longitud = convertir_longitud(longitud_val, longitud_unidad)
        diametro = convertir_longitud(diametro_val, diametro_unidad)
        rugosidad = convertir_longitud(rugosidad_val, rugosidad_unidad)
        altura_geodesica = convertir_longitud(altura_geodesica_val, altura_unidad)
        
        # Obtener datos
        Q_data = edited_df.iloc[:, 0].values
        P1_data = edited_df.iloc[:, 1].values
        P2_data = edited_df.iloc[:, 2].values
        
        # Convertir unidades de los datos
        Q = np.array([convertir_caudal(q, caudal_unidad) for q in Q_data])
        P1 = np.array([convertir_presion(p, presion_unidad_in) for p in P1_data])
        P2 = np.array([convertir_presion(p, presion_unidad_out) for p in P2_data])
        
        # Validación de presiones
        if any(p2 < p1 for p1, p2 in zip(P1, P2)):
            st.warning("Advertencia: Algunas presiones de salida son menores que las de entrada")
        
        # Calcular altura del sistema para cada punto medido
        H_sistema = np.zeros(len(Q))
        for i in range(len(Q)):
            perdidas = calcular_perdidas(Q[i], longitud, diametro, rugosidad, 
                                      coef_perdida_total, densidad, viscosidad)
            H_sistema[i] = (P2[i] - P1[i])/(densidad * 9.81) + altura_geodesica + perdidas
        
        # Ajustar curva a los puntos
        def modelo_curva(Q, a, b, c):
            return a + b*Q + c*Q**2
        
        popt, _ = curve_fit(modelo_curva, Q, H_sistema)
        
        # Generar curva suavizada
        Q_smooth = np.linspace(min(Q), max(Q), 100)
        H_smooth = modelo_curva(Q_smooth, *popt)
        
        # Calcular curva teórica
        H_teorico = [altura_sistema(q, altura_geodesica, longitud, diametro, 
                                   rugosidad, coef_perdida_total, 
                                   densidad, viscosidad) for q in Q_smooth]
        
        # Mostrar resultados
        st.subheader("Resultados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Parámetros de la curva ajustada:**")
            st.write(f"- Altura estática (a) = {popt[0]:.4f} m")
            st.write(f"- Término lineal (b) = {popt[1]:.4f} s/m²")
            st.write(f"- Término cuadrático (c) = {popt[2]:.4f} s²/m⁵")
            
            st.write("\n**Ecuación de la curva del sistema:**")
            st.write(f"H = {popt[0]:.4f} + {popt[1]:.4f}·Q + {popt[2]:.4f}·Q²")
            
            st.write("\n**Parámetros en unidades base:**")
            st.write(f"- Densidad: {densidad:.4f} kg/m³")
            st.write(f"- Viscosidad: {viscosidad:.6f} Pa·s")
            st.write(f"- Diámetro tubería: {diametro:.6f} m")
            st.write(f"- Rugosidad relativa: {rugosidad/diametro:.6f}")
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(Q, H_sistema, 'o', label='Datos medidos')
            ax.plot(Q_smooth, H_smooth, '-', label='Curva ajustada')
            ax.plot(Q_smooth, H_teorico, '--', label='Curva teórica')
            
            ax.set_xlabel(f'Caudal (m³/s)')
            ax.set_ylabel('Altura (m)')
            ax.set_title('Curva del Sistema de Bombeo')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend()
            
            st.pyplot(fig)
        
        # Mostrar tabla de resultados detallados
        st.subheader("Resultados Detallados por Punto")
        resultados = pd.DataFrame({
            'Q (m³/s)': Q,
            'Velocidad (m/s)': Q / (np.pi * (diametro/2)**2),
            'Número de Reynolds': (densidad * Q * diametro) / (viscosidad * np.pi * (diametro/2)**2),
            'Factor de fricción': [factor_friccion(Re, rugosidad/diametro) for Re in (densidad * Q * diametro) / (viscosidad * np.pi * (diametro/2)**2)],
            'Altura sistema (m)': H_sistema
        })
        st.dataframe(resultados.style.format({
            'Q (m³/s)': "{:.6f}",        # 8 decimales para caudal
            'Velocidad (m/s)': "{:.6f}",  # 6 decimales para velocidad
            'Número de Reynolds': "{:.2f}", 
            'Factor de fricción': "{:.6f}",
            'Altura sistema (m)': "{:.4f}"
        }))
        
    except Exception as e:
        st.error(f"Error al procesar los datos: {str(e)}")
        st.error("Verifique que todos los datos de entrada sean válidos")

# Explicación de mejoras
st.header("Mejoras Implementadas")
st.markdown("""
1. **Cálculo mejorado del factor de fricción**:
   - Considera flujo laminar (Re < 2000)
   - Zona de transición (2000 ≤ Re < 4000)
   - Flujo turbulento (Re ≥ 4000)

2. **Pérdidas secundarias más realistas**:
   - Permite diferentes tipos de accesorios
   - Cada tipo puede tener su propio coeficiente K

3. **Validación de datos mejorada**:
   - Verifica caudales no negativos
   - Comprueba que el diámetro sea positivo
   - Valida relaciones de presión entrada/salida

4. **Mayor precisión en cálculos**:
   - Más decimales en resultados
   - Unidades consistentes en todos los cálculos

5. **Resultados más completos**:
   - Tabla detallada por punto de operación
   - Visualización de parámetros intermedios
""")