import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Configuración de la página
st.set_page_config(page_title="Análisis de Sistema de Bombeo", layout="wide")
st.title("📊 Análisis Completo de Sistema de Bombeo")

# =============================================
# FUNCIONES DE CONVERSIÓN DE UNIDADES
# =============================================

def convertir_presion(valor, unidad_in, unidad_out='Pa'):
    """Convierte entre unidades de presión"""
    factores = {
        'Pa': 1, 'kPa': 1000, 'bar': 100000,
        'psi': 6894.76, 'inHg': 3386.39, 'mmHg': 133.322
    }
    return valor * factores[unidad_in] / factores[unidad_out]

def convertir_longitud(valor, unidad_in, unidad_out='m'):
    """Convierte entre unidades de longitud"""
    factores = {
        'm': 1, 'cm': 0.01, 'mm': 0.001,
        'ft': 0.3048, 'in': 0.0254
    }
    return valor * factores[unidad_in] / factores[unidad_out]

def convertir_caudal(valor, unidad_in, unidad_out='m³/s'):
    """Convierte entre unidades de caudal"""
    factores = {
        'm³/s': 1, 'l/min': 0.001/60,
        'm³/h': 1/3600, 'gal/min': 6.30902e-5
    }
    return valor * factores[unidad_in] / factores[unidad_out]

def convertir_densidad(valor, unidad_in, unidad_out='kg/m³'):
    """Convierte entre unidades de densidad"""
    factores = {
        'kg/m³': 1, 'g/cm³': 1000, 'lb/ft³': 16.0185
    }
    return valor * factores[unidad_in] / factores[unidad_out]

def convertir_viscosidad(valor, unidad_in, unidad_out='Pa·s'):
    """Convierte entre unidades de viscosidad"""
    factores_in = {'Pa·s': 1, 'cP': 0.001, 'lb/(ft·s)': 1.48816}
    factores_out = {'Pa·s': 1, 'cP': 1000, 'lb/(ft·s)': 0.67197}
    return valor * factores_in[unidad_in] / factores_out[unidad_out]

# =============================================
# FUNCIONES DE CÁLCULO HIDRÁULICO
# =============================================

def factor_friccion(Re, rugosidad_relativa):
    """Calcula el factor de fricción usando la aproximación de Haaland"""
    if Re < 2000:
        return 64 / max(Re, 1e-6)  # Evitar división por cero
    elif 2000 <= Re < 4000:
        f_lam = 64 / 2000
        f_turb = (-1.8 * np.log10((6.9/4000) + (rugosidad_relativa/3.7)**1.11))**(-2)
        return f_lam + (f_turb - f_lam) * ((Re - 2000) / 2000)
    else:
        return (-1.8 * np.log10((6.9/Re) + (rugosidad_relativa/3.7)**1.11))**(-2)

def calcular_componentes(Q, L, D, rugosidad, coef_perdida_total, densidad, viscosidad):
    """Calcula todos los componentes hidráulicos para un caudal dado"""
    A = np.pi * (D**2) / 4
    v = Q / A
    Re = (densidad * v * D) / max(viscosidad, 1e-6)  # Evitar división por cero
    rug_rel = rugosidad / D
    
    if Re == 0:
        return 0, 0, 0, 0, 0
    
    f = factor_friccion(Re, rug_rel)
    hf_prim = f * (L/D) * (v**2) / (2 * 9.81)
    hf_sec = coef_perdida_total * (v**2) / (2 * 9.81)
    
    return v, Re, f, hf_prim, hf_sec

# =============================================
# INTERFAZ DE USUARIO - ENTRADA DE DATOS
# =============================================

with st.sidebar:
    st.header("⚙️ Parámetros del Sistema")
    
    # Propiedades del fluido
    st.subheader("💧 Propiedades del Fluido")
    col1, col2 = st.columns(2)
    with col1:
        densidad_val = st.number_input("Densidad", value=998.2, format="%.2f")
        densidad_unidad = st.selectbox("Unidad densidad", ['kg/m³', 'g/cm³', 'lb/ft³'])
    with col2:
        viscosidad_val = st.number_input("Viscosidad dinámica", value=1.002, format="%.4f")
        viscosidad_unidad = st.selectbox("Unidad viscosidad", ['cP', 'Pa·s', 'lb/(ft·s)'])
    
    # Geometría de la tubería
    st.subheader("📏 Geometría de la Tubería")
    col1, col2 = st.columns(2)
    with col1:
        longitud_val = st.number_input("Longitud total", value=100.0)
        longitud_unidad = st.selectbox("Unidad longitud", ['m', 'ft', 'in', 'cm', 'mm'])
    with col2:
        diametro_val = st.number_input("Diámetro interno", value=0.1)
        diametro_unidad = st.selectbox("Unidad diámetro", ['m', 'ft', 'in', 'cm', 'mm'])
    
    col1, col2 = st.columns(2)
    with col1:
        rugosidad_val = st.number_input("Rugosidad absoluta", value=0.045, format="%.4f")
    with col2:
        rugosidad_unidad = st.selectbox("Unidad rugosidad", ['mm', 'm', 'ft', 'in'])
    
    # Altura geodésica
    st.subheader("📐 Altura Geodésica")
    col1, col2 = st.columns(2)
    with col1:
        altura_geodesica_val = st.number_input("Diferencia de altura", value=10.0)
    with col2:
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

# Datos de caudal y presión
st.header("📈 Datos de Operación")
st.write("Ingrese los datos medidos de caudal y presión:")

col1, col2, col3 = st.columns(3)
with col1:
    caudal_unidad = st.selectbox("Unidad caudal", ['m³/s', 'l/min', 'm³/h', 'gal/min'])
with col2:
    presion_unidad_in = st.selectbox("Unidad presión entrada", ['Pa', 'kPa', 'bar', 'psi', 'inHg', 'mmHg'])
with col3:
    presion_unidad_out = st.selectbox("Unidad presión salida", ['Pa', 'kPa', 'bar', 'psi', 'inHg', 'mmHg'])

# Crear DataFrame editable
data = {
    f'Q ({caudal_unidad})': [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
    f'P1 ({presion_unidad_in})': [0]*10 if presion_unidad_in != 'inHg' else [29.92]*10,
    f'P2 ({presion_unidad_out})': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] if presion_unidad_out != 'inHg' else [29.92, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0]
}
df = pd.DataFrame(data)
edited_df = st.data_editor(df, num_rows="dynamic", height=300)

# =============================================
# PROCESAMIENTO DE DATOS
# =============================================

if st.button("🔄 Calcular Curva del Sistema", type="primary"):
    try:
        # Validación básica
        if any(q < 0 for q in edited_df.iloc[:, 0]):
            st.error("❌ Error: Los caudales no pueden ser negativos")
            st.stop()
            
        if diametro_val <= 0:
            st.error("❌ Error: El diámetro debe ser positivo")
            st.stop()
            
        # Conversión de unidades
        densidad = convertir_densidad(densidad_val, densidad_unidad)
        viscosidad = convertir_viscosidad(viscosidad_val, viscosidad_unidad)
        longitud = convertir_longitud(longitud_val, longitud_unidad)
        diametro = convertir_longitud(diametro_val, diametro_unidad)
        rugosidad = convertir_longitud(rugosidad_val, rugosidad_unidad)
        altura_geodesica = convertir_longitud(altura_geodesica_val, altura_unidad)
        
        # Obtener y convertir datos medidos
        Q_data = edited_df.iloc[:, 0].values
        P1_data = edited_df.iloc[:, 1].values
        P2_data = edited_df.iloc[:, 2].values
        
        Q = np.array([convertir_caudal(q, caudal_unidad) for q in Q_data])
        P1 = np.array([convertir_presion(p, presion_unidad_in) for p in P1_data])
        P2 = np.array([convertir_presion(p, presion_unidad_out) for p in P2_data])
        
        # Validación de presiones
        if any(p2 < p1 for p1, p2 in zip(P1, P2)):
            st.warning("⚠️ Advertencia: Algunas presiones de salida son menores que las de entrada")
        
        # Inicializar arrays para resultados
        resultados = {
            'Q (m³/s)': Q,
            'Velocidad (m/s)': np.zeros(len(Q)),
            'Número de Reynolds': np.zeros(len(Q)),
            'Factor de fricción': np.zeros(len(Q)),
            'Altura presión (m)': (P2 - P1)/(densidad * 9.81),
            'Pérdidas primarias (m)': np.zeros(len(Q)),
            'Pérdidas secundarias (m)': np.zeros(len(Q)),
            'Altura sistema (m)': np.zeros(len(Q))
        }
        
        # Calcular para cada punto
        for i in range(len(Q)):
            v, Re, f, hf_prim, hf_sec = calcular_componentes(
                Q[i], longitud, diametro, rugosidad,
                coef_perdida_total, densidad, viscosidad
            )
            
            resultados['Velocidad (m/s)'][i] = v
            resultados['Número de Reynolds'][i] = Re
            resultados['Factor de fricción'][i] = f
            resultados['Pérdidas primarias (m)'][i] = hf_prim
            resultados['Pérdidas secundarias (m)'][i] = hf_sec
            resultados['Altura sistema (m)'][i] = (
                resultados['Altura presión (m)'][i] + 
                altura_geodesica + 
                hf_prim + 
                hf_sec
            )
        
        # Convertir a DataFrame
        df_resultados = pd.DataFrame(resultados)
        
        # Ajustar curva a los puntos
        def modelo_curva(Q, a, b, c):
            return a + b*Q + c*Q**2
        
        popt, _ = curve_fit(modelo_curva, Q, resultados['Altura sistema (m)'])
        
        # Generar curva suavizada
        Q_smooth = np.linspace(min(Q), max(Q), 100)
        H_smooth = modelo_curva(Q_smooth, *popt)
        
        # Calcular curva teórica
        H_teorico = [
            resultados['Altura presión (m)'][0] + altura_geodesica +
            calcular_componentes(q, longitud, diametro, rugosidad,
                               coef_perdida_total, densidad, viscosidad)[3] +
            calcular_componentes(q, longitud, diametro, rugosidad,
                               coef_perdida_total, densidad, viscosidad)[4]
            for q in Q_smooth
        ]
        
        # =============================================
        # VISUALIZACIÓN DE RESULTADOS
        # =============================================
        
        st.header("📊 Resultados del Análisis")
        
        # Gráfico de la curva del sistema
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(Q, resultados['Altura sistema (m)'], 'o', label='Datos medidos', markersize=8)
        ax.plot(Q_smooth, H_smooth, '-', label='Curva ajustada', linewidth=2)
        ax.plot(Q_smooth, H_teorico, '--', label='Curva teórica', linewidth=2)
        
        ax.set_xlabel(f'Caudal ({caudal_unidad})', fontsize=12)
        ax.set_ylabel('Altura (m)', fontsize=12)
        ax.set_title('Curva Característica del Sistema', fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(fontsize=10)
        st.pyplot(fig)
        
        # Mostrar parámetros de la curva
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔧 Parámetros de la Curva")
            st.markdown(f"""
            - **Altura estática (a):** `{popt[0]:.4f} m`
            - **Coeficiente lineal (b):** `{popt[1]:.4f} s/m²`
            - **Coeficiente cuadrático (c):** `{popt[2]:.6f} s²/m⁵`
            
            **Ecuación de la curva:**  
            `H = {popt[0]:.4f} + {popt[1]:.4f}·Q + {popt[2]:.6f}·Q²`
            """)
            
        with col2:
            st.subheader("⚙️ Parámetros del Sistema")
            st.markdown(f"""
            - **Densidad:** `{densidad:.4f} kg/m³`
            - **Viscosidad:** `{viscosidad:.6f} Pa·s`
            - **Diámetro tubería:** `{diametro:.4f} m`
            - **Rugosidad relativa:** `{rugosidad/diametro:.6f}`
            - **Coef. pérdidas total (ΣK):** `{coef_perdida_total:.2f}`
            """)
        
        # Mostrar tabla de resultados detallados
        st.subheader("📋 Resultados Detallados por Punto")
        st.dataframe(
            df_resultados.style.format({
                'Q (m³/s)': "{:.6f}",
                'Velocidad (m/s)': "{:.6f}",
                'Número de Reynolds': "{:.2f}",
                'Factor de fricción': "{:.6f}",
                'Altura presión (m)': "{:.4f}",
                'Pérdidas primarias (m)': "{:.4f}",
                'Pérdidas secundarias (m)': "{:.4f}",
                'Altura sistema (m)': "{:.4f}"
            }),
            height=400
        )
        
        # Opción para descargar resultados
        csv = df_resultados.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar resultados completos (CSV)",
            data=csv,
            file_name="resultados_sistema_bombeo.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ Error al procesar los datos: {str(e)}")
        st.error("Por favor verifique que todos los datos de entrada sean válidos")

# =============================================
# INFORMACIÓN ADICIONAL
# =============================================

with st.expander("📚 Teoría y Referencias"):
    st.markdown(r"""
    ## **Fundamento Teórico**
    
    ### 1. Pérdidas de Carga
    - **Primarias (por fricción):**  
      $$h_f = f\frac{L}{D}\frac{v^2}{2g}$$  
      Donde $f$ es el factor de fricción calculado con la ecuación de Haaland
    
    ### 2. Factor de Fricción (Haaland)
    $$ \frac{1}{\sqrt{f}} \approx -1.8 \log_{10}\left[\left(\frac{\epsilon/D}{3.7}\right)^{1.11} + \frac{6.9}{Re}\right] $$
    
    O en forma explícita:
    $$ f = \left(-1.8 \log_{10}\left[\left(\frac{\epsilon/D}{3.7}\right)^{1.11} + \frac{6.9}{Re}\right]\right)^{-2} $$
    
    ### 3. Rango de Aplicación
    - Flujo turbulento ($Re \geq 4000$)
    - Precisión: $\pm 2\%$ vs. Colebrook-White
    """)

with st.expander("ℹ️ Instrucciones de Uso"):
    st.markdown("""
    1. **Complete todos los parámetros** en la barra lateral
    2. **Ingrese los datos medidos** en la tabla principal
    3. **Haga clic en 'Calcular Curva del Sistema'**
    4. **Revise los resultados** en las secciones inferiores
    5. **Descargue los datos** si es necesario
    """)