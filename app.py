
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Rutas de los archivos
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_rendimiento_academico.pkl"
DATA_PATH = BASE_DIR / "StudentsPerformance.csv"

# Configuración de la página
st.set_page_config(
    page_title="Predicción del rendimiento académico",
    page_icon="🎓",
    layout="centered"
)

# Cargar modelo y dataset
@st.cache_resource
def cargar_modelo():
    return joblib.load(MODEL_PATH)

@st.cache_data
def cargar_datos():
    return pd.read_csv(DATA_PATH)

try:
    modelo = cargar_modelo()
    df = cargar_datos()
except FileNotFoundError as error:
    st.error(f"No se encontró el archivo: {error.filename}")
    st.stop()
except Exception as error:
    st.error(f"No fue posible cargar los archivos: {error}")
    st.stop()

# Interfaz
st.title("🎓 Predicción del rendimiento académico")
st.write(
    "Ingresa la información del estudiante para estimar "
    "su promedio académico."
)

st.info(
    "La estimación se basa en un modelo de Machine Learning "
    "con capacidad predictiva limitada."
)

# Obtener opciones desde el dataset
gender_options = sorted(df["gender"].dropna().unique().tolist())
race_options = sorted(df["race/ethnicity"].dropna().unique().tolist())
education_options = sorted(
    df["parental level of education"].dropna().unique().tolist()
)
lunch_options = sorted(df["lunch"].dropna().unique().tolist())
preparation_options = sorted(
    df["test preparation course"].dropna().unique().tolist()
)

# Formulario
with st.form("student_form"):
    gender = st.selectbox("Género", gender_options)

    race = st.selectbox("Grupo étnico", race_options)

    parental_education = st.selectbox(
        "Nivel educativo de los padres",
        education_options
    )

    lunch = st.selectbox("Tipo de almuerzo", lunch_options)

    test_preparation = st.selectbox(
        "Curso de preparación para el examen",
        preparation_options
    )

    submitted = st.form_submit_button("Estimar promedio")

# Predicción
if submitted:
    estudiante = pd.DataFrame([{
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": parental_education,
        "lunch": lunch,
        "test preparation course": test_preparation
    }])

    try:
        prediccion = float(modelo.predict(estudiante)[0])
        prediccion = max(0.0, min(100.0, prediccion))

        st.success(
            f"Promedio académico estimado: {prediccion:.2f}/100"
        )

        st.caption(
            "El resultado es una estimación estadística, no una "
            "garantía del rendimiento real del estudiante."
        )

    except Exception as error:
        st.error(f"No fue posible generar la predicción: {error}")

st.divider()
st.caption("Proyecto ACA 1 · Asignatura Aprendizaje Automático")
