import streamlit as st
import pandas as pd
import pickle


# =========================================================
# CONFIGURACIÓN DE LA PÁGINA
# =========================================================

st.set_page_config(
    page_title="Predicción de Metástasis",
    page_icon="🎗️",
    layout="centered"
)


# =========================================================
# TÍTULO
# =========================================================

st.title("🎗️ Predicción del Tipo de Metástasis")

st.write(
    "Ingrese los datos solicitados para realizar la predicción. "
    "Las variables socioeconómicas se obtienen automáticamente "
    "a partir del código ZIP3."
)


# =========================================================
# CARGAR MODELO
# =========================================================

@st.cache_resource
def cargar_modelo():

    with open("modelo-class.pkl", "rb") as archivo:
        modelo = pickle.load(archivo)

    return modelo


modelo = cargar_modelo()


# =========================================================
# CARGAR ARCHIVO DE ZIP3
# =========================================================

@st.cache_data
def cargar_datos_zip():

    datos = pd.read_csv("zip_completo.csv")

    return datos


zip_data = cargar_datos_zip()


# =========================================================
# VARIABLES QUE UTILIZA EL MODELO
# =========================================================

variables_modelo = [
    'payer_type',
    'patient_age',
    'breast_cancer_diagnosis_code',
    'population',
    'density',
    'age_median',
    'age_60s',
    'age_70s',
    'married',
    'never_married',
    'family_size',
    'income_household_median',
    'income_household_10_to_15',
    'income_household_75_to_100',
    'income_household_100_to_150',
    'income_household_150_over',
    'income_household_six_figure',
    'income_individual_median',
    'home_ownership',
    'housing_units',
    'rent_median',
    'rent_burden',
    'education_less_highschool',
    'education_highschool',
    'education_bachelors',
    'education_college_or_above',
    'labor_force_participation',
    'unemployment_rate',
    'self_employed',
    'farmer',
    'race_white',
    'race_other',
    'race_multiple',
    'hispanic',
    'disabled',
    'poverty',
    'limited_english',
    'veteran',
    'Ozone'
]


# =========================================================
# CATEGORÍAS DE PAYER TYPE
# =========================================================

payer_options = [
    "MEDICAID",
    "COMMERCIAL",
    "MEDICARE ADVANTAGE",
    "?"
]


# =========================================================
# CÓDIGOS DE DIAGNÓSTICO
# =========================================================

diagnostico_options = [
    'C50919',
    'C50411',
    'C50112',
    'C50212',
    '1749',
    'C50912',
    'C50512',
    '1744',
    'C50412',
    'C50812',
    'C50911',
    'C50312',
    'C50311',
    'C50111',
    '1741',
    'C5091',
    'C50811',
    '1748',
    'C50511',
    '1743',
    'C50211',
    'C50011',
    'C5051',
    'C50012',
    'C50419',
    '1742',
    'C50611',
    'C50612',
    'C50119',
    'C50819',
    '1746',
    'C5041',
    'C50619',
    '19881',
    'C5081',
    '1745',
    'C50219',
    'C50319',
    'C50019',
    'C50519',
    'C50929',
    'C50021',
    'C5021',
    'C5011',
    'C5031',
    'C509',
    'C50',
    '1759',
    'C5001',
    'C50421',
    'C50922',
    'C50921'
]


# =========================================================
# INTERFAZ
# =========================================================

st.subheader("📋 Datos del paciente")


zip3 = st.number_input(
    "Código ZIP3",
    min_value=0,
    max_value=999,
    value=100,
    step=1,
    help="Ingrese los primeros tres dígitos del código postal."
)


edad = st.number_input(
    "Edad del paciente",
    min_value=0,
    max_value=120,
    value=50,
    step=1
)


payer_type = st.selectbox(
    "Tipo de pagador",
    payer_options
)


diagnostico = st.selectbox(
    "Código de diagnóstico de cáncer de mama",
    diagnostico_options
)


# =========================================================
# BOTÓN DE PREDICCIÓN
# =========================================================

if st.button(
    "🔮 Realizar predicción",
    use_container_width=True
):

    # -----------------------------------------------------
    # BUSCAR EL ZIP3
    # -----------------------------------------------------

    registro_zip = zip_data[
        zip_data["patient_zip3"] == zip3
    ]


    # -----------------------------------------------------
    # VALIDAR ZIP3
    # -----------------------------------------------------

    if registro_zip.empty:

        st.error(
            f"El ZIP3 {zip3} no se encuentra en la base de datos."
        )

        st.stop()


    # -----------------------------------------------------
    # OBTENER REGISTRO DEL ZIP3
    # -----------------------------------------------------

    datos_zip = registro_zip.iloc[0]


    # -----------------------------------------------------
    # VERIFICAR DATOS FALTANTES
    # -----------------------------------------------------

    variables_zip = [
        variable
        for variable in variables_modelo
        if variable not in [
            'payer_type',
            'patient_age',
            'breast_cancer_diagnosis_code'
        ]
    ]


    datos_faltantes = [
        variable
        for variable in variables_zip
        if pd.isna(datos_zip[variable])
    ]


    if datos_faltantes:

        st.error(
            "El ZIP3 seleccionado contiene valores faltantes "
            "en algunas variables necesarias para la predicción."
        )

        st.write(
            "Variables faltantes:"
        )

        st.write(datos_faltantes)

        st.stop()


    # -----------------------------------------------------
    # CONSTRUIR REGISTRO PARA EL MODELO
    # -----------------------------------------------------

    datos_prediccion = {}


    for variable in variables_modelo:

        # Variable ingresada por el usuario

        if variable == "payer_type":

            datos_prediccion[variable] = payer_type


        elif variable == "patient_age":

            datos_prediccion[variable] = edad


        elif variable == "breast_cancer_diagnosis_code":

            datos_prediccion[variable] = diagnostico


        # Variables obtenidas mediante ZIP3

        else:

            datos_prediccion[variable] = datos_zip[variable]


    # -----------------------------------------------------
    # CREAR DATAFRAME
    # -----------------------------------------------------

    X_nuevo = pd.DataFrame(
        [datos_prediccion],
        columns=variables_modelo
    )


    # -----------------------------------------------------
    # REALIZAR PREDICCIÓN
    # -----------------------------------------------------

    prediccion = modelo.predict(X_nuevo)[0]


    # -----------------------------------------------------
    # CONVERTIR PREDICCIÓN A NOMBRE
    # -----------------------------------------------------

    nombres_clases = {
        0: "Esqueletico",
        1: "Linfatico",
        2: "VisceralSistemico"
    }


    try:

        resultado = nombres_clases[int(prediccion)]

    except (ValueError, KeyError):

        resultado = str(prediccion)


    # =====================================================
    # MOSTRAR RESULTADO
    # =====================================================

    st.success("Predicción realizada correctamente.")

    st.subheader("🔬 Resultado")

    st.info(
        f"**Tipo de metástasis predicho:** {resultado}"
    )


    # =====================================================
    # INFORMACIÓN UTILIZADA
    # =====================================================

    st.subheader("📍 Información del ZIP3")

    col1, col2 = st.columns(2)


    with col1:

        st.write(
            f"**ZIP3:** {zip3}"
        )

        st.write(
            f"**Edad:** {edad}"
        )


    with col2:

        st.write(
            f"**Población:** {datos_zip['population']:,.0f}"
        )

        st.write(
            f"**Densidad:** {datos_zip['density']:,.2f}"
        )


    # =====================================================
    # DATOS INGRESADOS
    # =====================================================

    st.subheader("📋 Datos ingresados")

    st.write(
        f"**Tipo de pagador:** {payer_type}"
    )

    st.write(
        f"**Código de diagnóstico:** {diagnostico}"
    )