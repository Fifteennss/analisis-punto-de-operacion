import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import fsolve

# Configuración de la página
st.set_page_config(page_title="Análisis de Sistema de Bombeo", layout="wide")
st.title("📊 Análisis Bomba-Sistema")

# =============================================
# FUNCIONES DE CONVERSIÓN DE UNIDADES
# =============================================
def convertir_presion(valor, unidad_in, unidad_out='Pa'):
    factores = {'Pa': 1, 'kPa': 1000, 'bar': 100000, 'psi': 6894.76, 'inHg': 3386.39, 'mmHg': 133.322}
    return valor * factores[unidad_in] / factores[unidad_out]

def convertir_longitud(valor, unidad_in, unidad_out='m'):
    factores = {'m': 1, 'cm': 0.01, 'mm': 0.001, 'ft': 0.3048, 'in': 0.0254}
    return valor * factores[unidad_in] / factores[unidad_out]

def convertir_caudal(valor, unidad_in, unidad_out='m³/s'):
    factores = {'m³/s': 1, 'l/min': 0.001/60, 'm³/h': 1/3600, 'gal/min': 6.30902e-5}
    return valor * factores[unidad_in] / factores[unidad_out]

def convertir_densidad(valor, unidad_in, unidad_out='kg/m³'):
    factores = {'kg/m³': 1, 'g/cm³': 1000, 'lb/ft³': 16.0185}
    return valor * factores[unidad_in] / factores[unidad_out]

def convertir_viscosidad(valor, unidad_in, unidad_out='Pa·s'):
    factores_in = {'Pa·s': 1, 'cP': 0.001, 'lb/(ft·s)': 1.48816}
    factores_out = {'Pa·s': 1, 'cP': 1000, 'lb/(ft·s)': 0.67197}
    return valor * factores_in[unidad_in] / factores_out[unidad_out]

# =============================================
# FUNCIONES DE CÁLCULO HIDRÁULICO
# =============================================
def factor_friccion(Re, rugosidad_relativa):
    """Calcula el factor de fricción usando la aproximación de Haaland"""
    if Re < 2000:
        return 64 / max(Re, 1e-6)
    elif 2000 <= Re < 4000:
        f_lam = 64 / 2000
        f_turb = (-1.8 * np.log10((6.9/4000) + (rugosidad_relativa/3.7)**1.11)**(-2))
        return f_lam + (f_turb - f_lam) * ((Re - 2000) / 2000)
    else:
        return (-1.8 * np.log10((6.9/Re) + (rugosidad_relativa/3.7)**1.11))**(-2)

def calcular_perdidas(Q, L, D, rugosidad, coef_perdida_total, densidad, viscosidad):
    """Calcula las pérdidas para un caudal dado"""
    A = np.pi * (D**2) / 4
    v = Q / A
    Re = (densidad * v * D) / max(viscosidad, 1e-6)
    rug_rel = rugosidad / D
    
    f = factor_friccion(Re, rug_rel)
    hf_prim = f * (L/D) * (v**2) / (2 * 9.81)
    hf_sec = coef_perdida_total * (v**2) / (2 * 9.81)
    
    return v, Re, f, hf_prim, hf_sec

def curva_bomba(Q, a, b, c):
    """Modelo cuadrático para curva de bomba"""
    return a - b*Q - c*Q**2

def curva_sistema(Q, H_estatica, K):
    """Modelo para curva del sistema"""
    return H_estatica + K*Q**2

# =============================================
# INTERFAZ DE USUARIO
# =============================================
with st.sidebar:
    st.header("⚙️ Parámetros del Sistema")
    
    # Propiedades del fluido
    st.subheader("💧 Propiedades del Fluido")
    col1, col2 = st.columns(2)
    with col1:
        densidad_val = st.number_input("Densidad", value=998.2, min_value=0.1, format="%.2f")
        densidad_unidad = st.selectbox("Unidad densidad", ['kg/m³', 'g/cm³', 'lb/ft³'])
    with col2:
        viscosidad_val = st.number_input("Viscosidad dinámica", value=1.002, min_value=0.0, format="%.4f")
        viscosidad_unidad = st.selectbox("Unidad viscosidad", ['cP', 'Pa·s', 'lb/(ft·s)'])
    
    # Geometría de la tubería
    st.subheader("📏 Geometría de la Tubería")
    col1, col2 = st.columns(2)
    with col1:
        longitud_val = st.number_input("Longitud total", value=100.0, min_value=0.1)
        longitud_unidad = st.selectbox("Unidad longitud", ['m', 'ft', 'in', 'cm', 'mm'])
    with col2:
        diametro_val = st.number_input("Diámetro interno", value=0.1, min_value=0.001)
        diametro_unidad = st.selectbox("Unidad diámetro", ['m', 'ft', 'in', 'cm', 'mm'])
    
    col1, col2 = st.columns(2)
    with col1:
        rugosidad_val = st.number_input("Rugosidad absoluta", value=0.045, min_value=0.0, format="%.4f")
    with col2:
        rugosidad_unidad = st.selectbox("Unidad rugosidad", ['mm', 'm', 'ft', 'in'])
    
    # Altura geodésica
    st.subheader("📐 Altura Geodésica")
    altura_geodesica_val = st.number_input("Diferencia de altura", value=10.0, min_value=0.0)
    altura_unidad = st.selectbox("Unidad altura", ['m', 'ft', 'in'])
    
    # Pérdidas secundarias
    st.subheader("🔩 Pérdidas Secundarias (Accesorios)")
    num_tipos = st.number_input("Número de tipos de accesorios", 1, 10, 1)
    coef_perdida_total = 0.0
    
    for i in range(num_tipos):
        st.markdown(f"**Accesorio tipo {i+1}**")
        col1, col2 = st.columns(2)
        with col1:
            num = st.number_input(f"Número", 0, 100, 1, key=f"num_{i}")
        with col2:
            k = st.number_input(f"Coeficiente K", 0.0, 15.0, 0.5, key=f"k_{i}")
        coef_perdida_total += num * k

# Datos de la bomba
st.header("📈 Datos de la Bomba")
st.write("Ingrese los datos de caudal y presiones de la bomba:")

# Convertir densidad antes de usar
densidad = convertir_densidad(densidad_val, densidad_unidad)

col1, col2, col3, col4 = st.columns(4)
with col1:
    caudal_unidad = st.selectbox("Unidad caudal", ['m³/s', 'l/min', 'm³/h', 'gal/min'])
with col2:
    presion_unidad_entrada = st.selectbox("Unidad presión entrada", ['Pa', 'kPa', 'bar', 'psi', 'inHg', 'mmHg'])
with col3:
    presion_unidad_salida = st.selectbox("Unidad presión salida", ['Pa', 'kPa', 'bar', 'psi', 'inHg', 'mmHg'])
with col4:
    altura_unidad_bomba = st.selectbox("Unidad altura calculada", ['m', 'ft'])

data = {
    f'Q ({caudal_unidad})': [0.0, 10.0, 20.0, 30.0, 40.0],
    f'P entrada ({presion_unidad_entrada})': [100.0, 100.0, 100.0, 100.0, 100.0],
    f'P salida ({presion_unidad_salida})': [394.0, 374.0, 344.0, 294.0, 214.0]
}
df = pd.DataFrame(data)
edited_df = st.data_editor(df, num_rows="dynamic", height=300)

# Calcular altura a partir de las presiones usando la densidad ya convertida
if len(edited_df) > 0:
    P_in = np.array([convertir_presion(p, presion_unidad_entrada, 'Pa') for p in edited_df.iloc[:, 1]])
    P_out = np.array([convertir_presion(p, presion_unidad_salida, 'Pa') for p in edited_df.iloc[:, 2]])
    H_calculada = (P_out - P_in) / (densidad * 9.81)  # Altura en metros

if st.button("🔄 Calcular Punto de Operación", type="primary"):
    try:
        # Validación básica
        if len(edited_df) < 3:
            st.error("Se necesitan al menos 3 puntos para la curva de la bomba")
            st.stop()
            
        if any(q < 0 for q in edited_df.iloc[:, 0]):
            st.error("❌ Los caudales no pueden ser negativos")
            st.stop()
            
        # Conversión de unidades
        densidad = convertir_densidad(densidad_val, densidad_unidad)
        viscosidad = convertir_viscosidad(viscosidad_val, viscosidad_unidad)
        longitud = convertir_longitud(longitud_val, longitud_unidad)
        diametro = convertir_longitud(diametro_val, diametro_unidad)
        rugosidad = convertir_longitud(rugosidad_val, rugosidad_unidad)
        altura_geodesica = convertir_longitud(altura_geodesica_val, altura_unidad)
        
        # Obtener y convertir datos de la bomba
        Q_bomba_data = edited_df.iloc[:, 0].values
        # Calcular H_bomba_data a partir de las presiones
        P_in = np.array([convertir_presion(p, presion_unidad_entrada, 'Pa') for p in edited_df.iloc[:, 1]])
        P_out = np.array([convertir_presion(p, presion_unidad_salida, 'Pa') for p in edited_df.iloc[:, 2]])
        H_bomba_data = (P_out - P_in) / (densidad * 9.81)  # Altura en metros
        
        Q_bomba = np.array([convertir_caudal(q, caudal_unidad) for q in Q_bomba_data])
        H_bomba = np.array([convertir_longitud(h, 'm', altura_unidad_bomba) for h in H_bomba_data])
        
        # Ajustar curva a los datos de la bomba
        popt_bomba, _ = curve_fit(curva_bomba, Q_bomba, H_bomba)
        
        # Calcular parámetros a caudal máximo
        Q_max = max(Q_bomba)
        v_max, Re_max, f_max, hf_prim_max, hf_sec_max = calcular_perdidas(
            Q_max, longitud, diametro, rugosidad, coef_perdida_total, densidad, viscosidad
        )
        
        # Calcular coeficiente K del sistema
        K_sistema = (hf_prim_max + hf_sec_max) / (Q_max**2)
        H_estatica = altura_geodesica
        
        # Encontrar punto de operación
        def ecuaciones(Q):
            return curva_bomba(Q, *popt_bomba) - curva_sistema(Q, H_estatica, K_sistema)
        
        Q_op = fsolve(ecuaciones, Q_max/2)[0]
        H_op = curva_bomba(Q_op, *popt_bomba)
        
        # Validar intersección
        if Q_op < 0 or Q_op > max(Q_bomba)*1.1:
            st.error("¡No hay punto de operación dentro del rango válido!")
            st.stop()
        
        # =============================================
        # VISUALIZACIÓN DE RESULTADOS
        # =============================================
        st.header("📊 Resultados del Análisis")
        
        # Generar curvas para graficar
        Q_range = np.linspace(0, max(Q_bomba)*1.1, 100)
        H_bomba_curve = curva_bomba(Q_range, *popt_bomba)
        H_sistema_curve = curva_sistema(Q_range, H_estatica, K_sistema)
        
        # Gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(Q_bomba, H_bomba, 'bo', label='Datos bomba', markersize=6)
        ax.plot(Q_range, H_bomba_curve, 'b-', label='Curva de la bomba', linewidth=2)
        ax.plot(Q_range, H_sistema_curve, 'r-', label='Curva del sistema', linewidth=2)
        ax.plot(Q_op, H_op, 'ro', markersize=8, label=f'Punto de operación (Q={Q_op:.3f} m³/s)')
        
        ax.set_xlabel(f'Caudal (m³/s)', fontsize=12)
        ax.set_ylabel('Altura (m)', fontsize=12)
        ax.set_title('Punto de Operación del Sistema', fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=10)
        st.pyplot(fig)
        
        # Resultados numéricos
        st.subheader("🔍 Resultados Numéricos")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📌 A Caudal Máximo**")
            st.markdown(f"""
            - Velocidad: `{v_max:.4f} m/s`
            - Número de Reynolds: `{Re_max:.2e}`
            - Factor de fricción: `{f_max:.6f}`
            - Pérdidas primarias: `{hf_prim_max:.4f} m`
            - Pérdidas secundarias: `{hf_sec_max:.4f} m`
            """)
            
        with col2:
            st.markdown("**📊 Ecuaciones**")
            st.markdown(f"""
            **Curva de la Bomba:**  
            `H = {popt_bomba[0]:.4f} - {popt_bomba[1]:.4f}·Q - {popt_bomba[2]:.6f}·Q²`
            
            **Curva del Sistema:**  
            `H = {H_estatica:.4f} + {K_sistema:.4f}·Q²`
            """)
        
        st.subheader("🎯 Punto de Operación")
        st.markdown(f"""
        - **Caudal:** `{Q_op:.6f} m³/s` (`{convertir_caudal(Q_op, 'm³/s', caudal_unidad):.2f} {caudal_unidad}`)
        - **Altura:** `{H_op:.4f} m` (`{convertir_longitud(H_op, 'm', altura_unidad_bomba):.2f} {altura_unidad_bomba}`)
        """)
        
    except Exception as e:
        st.error(f"❌ Error en los cálculos: {str(e)}")
        st.error("Verifique los datos de entrada y parámetros")

# Información adicional
with st.expander("📚 Teoría y Referencias"):
    st.markdown(r"""
    ## **Fundamento Teórico**
    
    ### 1. Curva Característica de la Bomba
    $$ H_{bomba} = a - bQ - cQ^2 $$
    
    ### 2. Curva del Sistema
    $$ H_{sistema} = H_{estatica} + KQ^2 $$
    Donde:
    $$ K = \frac{fL}{2gD(\pi D^2/4)^2} + \frac{\sum K}{2g(\pi D^2/4)^2} $$
    
    ### 3. Punto de Operación
    Intersección entre ambas curvas, donde:
    $$ H_{bomba}(Q_{op}) = H_{sistema}(Q_{op}) $$
    """)

# Pie de página de autoría
st.markdown("""
---
**Creado por:** Diego Gonzales Chapoñan  
Estudiante de Ingeniería Mecánica  
Universidad Nacional del Santa, 2025
""")