import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Heart Disease AI Dashboard", layout="wide")

DATA_FILE = "heart.csv"

# SIEMPRE RECARGA DATASET
df = pd.read_csv(DATA_FILE)

st.title("Heart Disease AI Dashboard")

# =========================
# SIDEBAR
# =========================
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Analytics", "Prediction + Add Patient"]
)

# =========================
# MODEL
# =========================
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# =========================
# OVERVIEW
# =========================
if page == "Overview":

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Patients", df.shape[0])
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Model Accuracy", f"{accuracy:.2f}")

    st.dataframe(df)

# =========================
# ANALYTICS
# =========================
elif page == "Analytics":

    st.subheader("Data Analysis Dashboard")

    fig1 = px.pie(
        df,
        names="target",
        title="Disease Distribution",
        color_discrete_sequence=["green", "red"]
    )
    st.plotly_chart(fig1, use_container_width=True)

    counts = df["target"].value_counts().reset_index()
    counts.columns = ["Class", "Count"]

    fig2 = px.bar(
        counts,
        x="Class",
        y="Count",
        color="Class",
        color_discrete_sequence=["green", "red"],
        title="Patient Distribution"
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        df,
        x="age",
        y="chol",
        color="target",
        title="Age vs Cholesterol Risk"
    )
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.imshow(
        df.corr(),
        text_auto=False,
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix"
    )
    st.plotly_chart(fig4, use_container_width=True)

# =========================
# PREDICTION + ADD PATIENT
# =========================
elif page == "Prediction + Add Patient":

    st.subheader("Patient Information Form")

    age = st.number_input("Age (Edad del paciente)", 1, 120, 50)

    sex = st.selectbox(
        "Sex (Sexo)",
        ["Female (0)", "Male (1)"]
    )

    cp = st.selectbox(
        "Chest Pain Type (Tipo de dolor en el pecho)",
        [
            "0 - Typical angina (dolor relacionado al corazón)",
            "1 - Atypical angina (dolor raro)",
            "2 - Non-anginal pain (no cardíaco)",
            "3 - Asymptomatic (sin síntomas)"
        ]
    )

    trestbps = st.number_input("Resting Blood Pressure (Presión arterial)", 80, 200, 120)
    chol = st.number_input("Cholesterol (Colesterol)", 100, 600, 200)

    fbs = st.selectbox(
        "Fasting Blood Sugar (Azúcar en sangre)",
        ["0 - Normal", "1 - Alto"]
    )

    restecg = st.selectbox(
        "ECG (Electrocardiograma)",
        ["0 - Normal", "1 - Alterado", "2 - Problemas cardíacos"]
    )

    thalach = st.number_input("Max Heart Rate (Frecuencia cardíaca máxima)", 60, 220, 150)

    exang = st.selectbox(
        "Exercise Angina (Dolor con ejercicio)",
        ["0 - No", "1 - Sí"]
    )

    oldpeak = st.number_input("Heart Stress Indicator (Oldpeak)", 0.0, 6.0, 1.0)

    slope = st.selectbox(
        "Heart Curve Slope (Estado del corazón)",
        ["0 - Mejor", "1 - Medio", "2 - Peor"]
    )

    ca = st.number_input("Blocked Vessels (Vasos bloqueados)", 0, 3, 0)

    thal = st.selectbox(
        "Thal (Condición del corazón)",
        ["1 - Normal", "2 - Defecto fijo", "3 - Defecto reversible"]
    )

    def parse(x):
        return int(x.split("-")[0].strip())

    input_data = np.array([[
        age,
        0 if "Female" in sex else 1,
        parse(cp),
        trestbps,
        chol,
        parse(fbs),
        parse(restecg),
        thalach,
        parse(exang),
        oldpeak,
        parse(slope),
        ca,
        parse(thal)
    ]])

    # =========================
    # PREDICT
    # =========================
    if st.button("Predict Risk"):
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        if pred == 1:
            st.error(f"High Risk of Heart Disease ({prob*100:.2f}%)")
        else:
            st.success(f"Low Risk ({(1-prob)*100:.2f}%)")

    # =========================
    # ADD PATIENT (ACTUALIZA DATASET)
    # =========================
    st.subheader("Add Patient to Database")

    target = st.selectbox("Diagnosis (Final result)", [0, 1])

    if st.button("Save Patient"):

        new_row = pd.DataFrame([[
            age,
            0 if "Female" in sex else 1,
            parse(cp),
            trestbps,
            chol,
            parse(fbs),
            parse(restecg),
            thalach,
            parse(exang),
            oldpeak,
            parse(slope),
            ca,
            parse(thal),
            target
        ]], columns=df.columns)

        df_updated = pd.concat([df, new_row], ignore_index=True)

        df_updated.to_csv(DATA_FILE, index=False)

        st.success("Patient saved successfully")

        st.rerun()