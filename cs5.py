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
    """Calcula el factor de fricción usando la aproximación de Haaland con transición mejorada"""
    Re = max(Re, 1e-6)  # Evitar división por cero
    
    if Re < 2000:
        return 64 / Re
    elif 2000 <= Re < 4000:
        # Interpolación lineal mejorada entre laminar y turbulento
        f_lam = 64 / 2000
        f_turb = (-1.8 * np.log10((6.9/4000) + (rugosidad_relativa/3.7)**1.11))**(-2)
        factor = (Re - 2000) / 2000
        return f_lam + (f_turb - f_lam) * factor
    else:
        return (-1.8 * np.log10((6.9/Re) + (rugosidad_relativa/3.7)**1.11))**(-2)

def calcular_perdidas(Q, L, D, rugosidad, coef_perdida_total, densidad, viscosidad):
    """Calcula las pérdidas para un caudal dado"""
    if Q <= 0:
        return 0, 0, 0, 0, 0
    
    A = np.pi * (D**2) / 4
    v = Q / A
    Re = (densidad * v * D) / max(viscosidad, 1e-6)
    rug_rel = rugosidad / max(D, 1e-6)
    
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
# FUNCIONES DE VALIDACIÓN
# =============================================
def validar_datos_bomba(edited_df, presion_unidad_entrada, presion_unidad_salida):
    """Valida los datos de entrada de la bomba"""
    errors = []
    warnings = []
    
    if len(edited_df) < 3:
        errors.append("Se necesitan al menos 3 puntos para ajustar la curva de la bomba")
        return errors, warnings
    
    # Verificar datos válidos
    for i, row in edited_df.iterrows():
        # Verificar valores no negativos para caudal
        if row.iloc[0] < 0:
            errors.append(f"Punto {i+1}: El caudal no puede ser negativo")
        
        # Verificar que P_salida > P_entrada
        p_in = convertir_presion(row.iloc[1], presion_unidad_entrada, 'Pa')
        p_out = convertir_presion(row.iloc[2], presion_unidad_salida, 'Pa')
        
        if p_out <= p_in:
            errors.append(f"Punto {i+1}: La presión de salida ({row.iloc[2]:.2f}) debe ser mayor que la de entrada ({row.iloc[1]:.2f})")
        
        # Advertencia si la diferencia de presión es muy baja
        delta_p = p_out - p_in
        if delta_p < 1000:  # Menos de 1 kPa
            warnings.append(f"Punto {i+1}: Diferencia de presión muy baja ({delta_p:.0f} Pa)")
    
    # Verificar monotonía en los datos (la altura debería disminuir con el caudal)
    caudales = sorted(edited_df.iloc[:, 0].values)
    if len(set(caudales)) != len(caudales):
        warnings.append("Hay caudales duplicados en los datos")
    
    return errors, warnings

def validar_parametros_geometricos(longitud, diametro, rugosidad, densidad, viscosidad):
    """Valida los parámetros geométricos del sistema"""
    errors = []
    warnings = []
    
    if longitud <= 0:
        errors.append("La longitud de la tubería debe ser positiva")
    
    if diametro <= 0:
        errors.append("El diámetro de la tubería debe ser positivo")
    
    if rugosidad < 0:
        errors.append("La rugosidad no puede ser negativa")
    
    if densidad <= 0:
        errors.append("La densidad debe ser positiva")
    
    if viscosidad <= 0:
        errors.append("La viscosidad debe ser positiva")
    
    # Advertencias
    if rugosidad/diametro > 0.05:
        warnings.append("Rugosidad relativa muy alta (>5%), verifique los valores")
    
    if diametro < 0.01:  # Menos de 1 cm
        warnings.append("Diámetro muy pequeño, verifique las unidades")
    
    if longitud > 10000:  # Más de 10 km
        warnings.append("Longitud muy grande, verifique las unidades")
    
    return errors, warnings

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

col1, col2, col3, col4 = st.columns(4)
with col1:
    caudal_unidad = st.selectbox("Unidad caudal", ['m³/s', 'l/min', 'm³/h', 'gal/min'])
with col2:
    presion_unidad_entrada = st.selectbox("Unidad presión entrada", ['Pa', 'kPa', 'bar', 'psi', 'inHg', 'mmHg'])
with col3:
    presion_unidad_salida = st.selectbox("Unidad presión salida", ['Pa', 'kPa', 'bar', 'psi', 'inHg', 'mmHg'])
with col4:
    altura_unidad_bomba = st.selectbox("Unidad altura mostrada", ['m', 'ft'])

data = {
    f'Q ({caudal_unidad})': [0.0, 10.0, 20.0, 30.0, 40.0],
    f'P entrada ({presion_unidad_entrada})': [100.0, 100.0, 100.0, 100.0, 100.0],
    f'P salida ({presion_unidad_salida})': [394.0, 374.0, 344.0, 294.0, 214.0]
}
df = pd.DataFrame(data)
edited_df = st.data_editor(df, num_rows="dynamic", height=300)

# Mostrar altura calculada para verificación
if len(edited_df) > 0:
    densidad_temp = convertir_densidad(densidad_val, densidad_unidad)
    st.subheader("🔍 Verificación - Altura Calculada")
    
    altura_data = []
    for i, row in edited_df.iterrows():
        P_in = convertir_presion(row.iloc[1], presion_unidad_entrada, 'Pa')
        P_out = convertir_presion(row.iloc[2], presion_unidad_salida, 'Pa')
        H_metros = (P_out - P_in) / (densidad_temp * 9.81)
        H_display = convertir_longitud(H_metros, 'm', altura_unidad_bomba)
        altura_data.append(H_display)
    
    df_verificacion = pd.DataFrame({
        f'Q ({caudal_unidad})': edited_df.iloc[:, 0].values,
        f'H calculada ({altura_unidad_bomba})': altura_data
    })
    st.dataframe(df_verificacion, use_container_width=True)

if st.button("🔄 Calcular Punto de Operación", type="primary"):
    try:
        # =============================================
        # VALIDACIONES EXHAUSTIVAS
        # =============================================
        
        # Convertir todas las unidades a SI
        densidad = convertir_densidad(densidad_val, densidad_unidad)
        viscosidad = convertir_viscosidad(viscosidad_val, viscosidad_unidad)
        longitud = convertir_longitud(longitud_val, longitud_unidad)
        diametro = convertir_longitud(diametro_val, diametro_unidad)
        rugosidad = convertir_longitud(rugosidad_val, rugosidad_unidad)
        altura_geodesica = convertir_longitud(altura_geodesica_val, altura_unidad)
        
        # Validar datos de la bomba
        errors_bomba, warnings_bomba = validar_datos_bomba(edited_df, presion_unidad_entrada, presion_unidad_salida)
        
        # Validar parámetros geométricos
        errors_geo, warnings_geo = validar_parametros_geometricos(longitud, diametro, rugosidad, densidad, viscosidad)
        
        # Mostrar errores críticos
        all_errors = errors_bomba + errors_geo
        if all_errors:
            st.error("❌ **Errores críticos encontrados:**")
            for error in all_errors:
                st.error(f"• {error}")
            st.stop()
        
        # Mostrar advertencias
        all_warnings = warnings_bomba + warnings_geo
        if all_warnings:
            st.warning("⚠️ **Advertencias:**")
            for warning in all_warnings:
                st.warning(f"• {warning}")
        
        # =============================================
        # PROCESAMIENTO DE DATOS
        # =============================================
        
        # Obtener y convertir datos de la bomba
        Q_bomba_data = edited_df.iloc[:, 0].values
        Q_bomba = np.array([convertir_caudal(q, caudal_unidad) for q in Q_bomba_data])
        
        # CORRECCIÓN CRÍTICA: Calcular altura correctamente
        P_in = np.array([convertir_presion(p, presion_unidad_entrada, 'Pa') for p in edited_df.iloc[:, 1]])
        P_out = np.array([convertir_presion(p, presion_unidad_salida, 'Pa') for p in edited_df.iloc[:, 2]])
        H_bomba_metros = (P_out - P_in) / (densidad * 9.81)  # Altura en metros
        
        # MANTENER EN METROS para todos los cálculos internos
        H_bomba = H_bomba_metros  # NO aplicar conversión adicional
        
        # Verificar que tenemos datos válidos para el ajuste
        if len(Q_bomba) < 3 or len(H_bomba) < 3:
            st.error("❌ Se necesitan al menos 3 puntos válidos para el análisis")
            st.stop()
        
        # Ajustar curva a los datos de la bomba
        try:
            popt_bomba, pcov_bomba = curve_fit(curva_bomba, Q_bomba, H_bomba)
            
            # Verificar que el ajuste es razonable
            r_squared = 1 - np.sum((H_bomba - curva_bomba(Q_bomba, *popt_bomba))**2) / np.sum((H_bomba - np.mean(H_bomba))**2)
            
            if r_squared < 0.8:
                st.warning(f"⚠️ Ajuste de curva pobre (R² = {r_squared:.3f}). Verifique los datos.")
                
        except Exception as e:
            st.error(f"❌ Error en el ajuste de la curva de la bomba: {str(e)}")
            st.error("Verifique que los datos sean consistentes y no tengan valores atípicos")
            st.stop()
        
        # =============================================
        # CÁLCULO DEL SISTEMA (MANTENIENDO MÉTODO ORIGINAL)
        # =============================================
        
        # Calcular parámetros a caudal máximo (MANTENIENDO MÉTODO ORIGINAL)
        Q_max = max(Q_bomba)
        v_max, Re_max, f_max, hf_prim_max, hf_sec_max = calcular_perdidas(
            Q_max, longitud, diametro, rugosidad, coef_perdida_total, densidad, viscosidad
        )
        
        # Calcular coeficiente K del sistema (MANTENIENDO MÉTODO ORIGINAL)
        if Q_max > 0:
            K_sistema = (hf_prim_max + hf_sec_max) / (Q_max**2)
        else:
            K_sistema = 0
            
        H_estatica = altura_geodesica
        
        # =============================================
        # ENCONTRAR PUNTO DE OPERACIÓN
        # =============================================
        
        def ecuaciones(Q):
            if Q <= 0:
                return float('inf')
            return curva_bomba(Q, *popt_bomba) - curva_sistema(Q, H_estatica, K_sistema)
        
        # Buscar punto de operación con mejor estimación inicial
        Q_inicial = Q_max * 0.7  # Estimación más conservadora
        
        try:
            Q_op = fsolve(ecuaciones, Q_inicial, xtol=1e-10)[0]
            H_op = curva_bomba(Q_op, *popt_bomba)
            
            # Validar que el punto de operación es físicamente razonable
            if Q_op <= 0:
                st.error("❌ No se encontró un punto de operación válido (caudal negativo)")
                st.stop()
            
            if Q_op > max(Q_bomba) * 1.5:
                st.warning("⚠️ El punto de operación está fuera del rango de datos de la bomba")
            
            if H_op <= 0:
                st.error("❌ Altura de operación negativa. Verifique los parámetros del sistema")
                st.stop()
                
        except Exception as e:
            st.error(f"❌ Error al encontrar el punto de operación: {str(e)}")
            st.error("Intente ajustar los parámetros del sistema o los datos de la bomba")
            st.stop()
        
        # =============================================
        # VISUALIZACIÓN DE RESULTADOS
        # =============================================
        st.header("📊 Resultados del Análisis")
        
        # Generar curvas para graficar
        Q_range = np.linspace(0, max(Q_bomba)*1.2, 100)
        H_bomba_curve = curva_bomba(Q_range, *popt_bomba)
        H_sistema_curve = curva_sistema(Q_range, H_estatica, K_sistema)
        
        # Gráfico mejorado
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Datos y curvas
        ax.plot(Q_bomba, H_bomba, 'bo', label='Datos bomba', markersize=8, markerfacecolor='lightblue', markeredgecolor='blue', markeredgewidth=2)
        ax.plot(Q_range, H_bomba_curve, 'b-', label='Curva de la bomba', linewidth=3)
        ax.plot(Q_range, H_sistema_curve, 'r-', label='Curva del sistema', linewidth=3)
        ax.plot(Q_op, H_op, 'ro', markersize=12, label=f'Punto de operación\n(Q={Q_op:.4f} m³/s, H={H_op:.2f} m)', 
                markerfacecolor='yellow', markeredgecolor='red', markeredgewidth=3)
        
        # Líneas de referencia
        ax.axhline(y=H_estatica, color='gray', linestyle='--', alpha=0.7, label=f'Altura estática ({H_estatica:.2f} m)')
        ax.axvline(x=Q_op, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=H_op, color='gray', linestyle=':', alpha=0.5)
        
        # Formato del gráfico
        ax.set_xlabel('Caudal (m³/s)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Altura (m)', fontsize=14, fontweight='bold')
        ax.set_title('Punto de Operación del Sistema Bomba-Tubería', fontsize=16, fontweight='bold')
        ax.grid(True, which='both', linestyle='-', alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        
        # Mejorar límites del gráfico
        ax.set_xlim(0, max(Q_bomba)*1.15)
        ax.set_ylim(0, max(max(H_bomba), H_op) * 1.15)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # =============================================
        # RESULTADOS NUMÉRICOS
        # =============================================
        st.subheader("🔍 Resultados Numéricos")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📌 Condiciones a Caudal Máximo**")
            st.markdown(f"""
            - **Velocidad:** `{v_max:.4f} m/s`
            - **Reynolds:** `{Re_max:.2e}`
            - **Factor fricción:** `{f_max:.6f}`
            - **Pérdidas primarias:** `{hf_prim_max:.4f} m`
            - **Pérdidas secundarias:** `{hf_sec_max:.4f} m`
            - **Pérdidas totales:** `{hf_prim_max + hf_sec_max:.4f} m`
            """)
            
        with col2:
            st.markdown("**📊 Ecuaciones del Sistema**")
            st.markdown(f"""
            **Curva de la Bomba:**  
            `H = {popt_bomba[0]:.4f} - {popt_bomba[1]:.6f}·Q - {popt_bomba[2]:.6f}·Q²`
            
            **Curva del Sistema:**  
            `H = {H_estatica:.4f} + {K_sistema:.4f}·Q²`
            
            **Coeficiente R²:** `{r_squared:.4f}`
            """)
            
        with col3:
            st.markdown("**🎯 Punto de Operación**")
            # Calcular condiciones en el punto de operación
            v_op, Re_op, f_op, hf_prim_op, hf_sec_op = calcular_perdidas(
                Q_op, longitud, diametro, rugosidad, coef_perdida_total, densidad, viscosidad
            )
            
            st.markdown(f"""
            - **Caudal:** `{Q_op:.6f} m³/s` ({convertir_caudal(Q_op, 'm³/s', caudal_unidad):.4f} {caudal_unidad})
            - **Altura:** `{H_op:.4f} m` ({convertir_longitud(H_op, 'm', altura_unidad_bomba):.4f} {altura_unidad_bomba})
            - **Velocidad:** `{v_op:.4f} m/s`
            - **Reynolds:** `{Re_op:.2e}`
            """)
        
        # =============================================
        # RESUMEN 
        # =============================================
        st.subheader("📋 Resumen")
        
        eficiencia_hidraulica = (H_estatica / H_op) * 100 if H_op > 0 else 0
        perdidas_porcentaje = ((hf_prim_op + hf_sec_op) / H_op) * 100 if H_op > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""
            **💡 Análisis de Eficiencia:**
            - Eficiencia hidráulica: **{eficiencia_hidraulica:.1f}%**
            - Pérdidas del sistema: **{perdidas_porcentaje:.1f}%**
            - Relación pérdidas prim./sec.: **{hf_prim_op/max(hf_sec_op, 1e-6):.2f}**
            """)
            
        with col2:
            # Clasificación del régimen de flujo
            if Re_op < 2000:
                regimen = "Laminar"
                color = "🔵"
            elif Re_op < 4000:
                regimen = "Transición"
                color = "🟡"
            else:
                regimen = "Turbulento"
                color = "🔴"
                
            st.success(f"""
            **🌊 Características del Flujo:**
            - Régimen: **{color} {regimen}**
            - Rugosidad relativa: **{(rugosidad/diametro)*100:.3f}%**
            - Velocidad recomendada: **1-3 m/s** ✓
            """)
        
        # Mostrar advertencias adicionales si es necesario
        if v_op > 3:
            st.warning("⚠️ Velocidad alta (>3 m/s): considere aumentar el diámetro para reducir pérdidas")
        elif v_op < 0.5:
            st.warning("⚠️ Velocidad muy baja (<0.5 m/s): riesgo de sedimentación")
            
        if eficiencia_hidraulica < 50:
            st.warning("⚠️ Eficiencia hidráulica baja: considere optimizar el diseño del sistema")
        
    except Exception as e:
        st.error(f"❌ Error inesperado en los cálculos: {str(e)}")
        st.error("**Posibles causas:**")
        st.error("• Datos de entrada inconsistentes")
        st.error("• Parámetros fuera de rangos válidos")
        st.error("• Error en las unidades seleccionadas")
        
        # Información de debug (opcional)
        with st.expander("🔧 Información de Debug"):
            st.write("**Parámetros convertidos:**")
            st.write(f"- Densidad: {densidad:.2f} kg/m³")
            st.write(f"- Viscosidad: {viscosidad:.6f} Pa·s")
            st.write(f"- Longitud: {longitud:.3f} m")
            st.write(f"- Diámetro: {diametro:.4f} m")
            st.write(f"- Rugosidad: {rugosidad:.6f} m")

# Información adicional
with st.expander("📚 Teoría y Referencias"):
    st.markdown(r"""
    ## **Fundamento Teórico**
    
    ### 1. Curva Característica de la Bomba
    $$ H_{bomba} = a - bQ - cQ^2 $$
    
    ### 2. Curva del Sistema
    $$ H_{sistema} = H_{estatica} + KQ^2 $$
    Donde:
    $$ K = \frac{hf_{primarias} + hf_{secundarias}}{Q_{max}^2} $$
    
    ### 3. Pérdidas Primarias (Darcy-Weisbach)
    $$ hf_{prim} = f \frac{L}{D} \frac{v^2}{2g} $$
    
    ### 4. Pérdidas Secundarias
    $$ hf_{sec} = \sum K \frac{v^2}{2g} $$
    
    ### 5. Factor de Fricción (Haaland)
    
    **Flujo Laminar (Re < 2000):**
    $ f = \frac{64}{Re} $
    
    **Flujo Turbulento (Re > 4000):**
    $ f = \left(-1.8 \log_{10}\left(\frac{6.9}{Re} + \left(\frac{\varepsilon/D}{3.7}\right)^{1.11}\right)\right)^{-2} $
    
    **Flujo de Transición (2000 ≤ Re ≤ 4000):**
    Interpolación lineal entre los valores laminar y turbulento.
    
    ### 6. Punto de Operación
    Intersección entre ambas curvas, donde:
    $ H_{bomba}(Q_{op}) = H_{sistema}(Q_{op}) $
    
    ### 7. Número de Reynolds
    $ Re = \frac{\rho v D}{\mu} = \frac{4 \rho Q}{\pi D \mu} $
    """)