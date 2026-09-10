import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Heavy Metal Adsorption AI",
    page_icon="💧",
    layout="wide"
)

DATA_FILE = "Dataset. S.Lamsiah.xlsx"
TARGET = "Adsorption capacity (mg/g)"

CAT_FEATURES = ["Heavy metal"]

NUM_FEATURES = [
    "Activation temperature (°C)",
    "Hydrated radius (nm)",
    "Electronegativity (Pauling)",
    "van der Waals radius (nm)",
    "Molar mass (g/mol)",
    "BET surface area (m²/g)",
    "Pore diameter (nm)",
    "Total pore volume (cm³/g)\n",
    "Temperature (°C)",
    "pH",
    "Dose (g/L)",
    "Contact time (min)",
    "Initial concentration (mg/L)",
]

FEATURES = CAT_FEATURES + NUM_FEATURES

# =========================================================
# DATA
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_excel(DATA_FILE)

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


# =========================================================
# MODEL
# =========================================================
@st.cache_resource
def train_model(df):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "metal",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT_FEATURES
            )
        ],
        remainder="passthrough"
    )

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "n_train": len(X_train),
        "n_test": len(X_test)
    }

    test_results = X_test.copy()
    test_results["Actual capacity (mg/g)"] = y_test.values
    test_results["Predicted capacity (mg/g)"] = y_pred
    test_results["Absolute error (mg/g)"] = np.abs(
        test_results["Actual capacity (mg/g)"] -
        test_results["Predicted capacity (mg/g)"]
    )

    return pipeline, metrics, test_results


def get_feature_importance(pipeline):
    pre = pipeline.named_steps["preprocessor"]
    rf = pipeline.named_steps["model"]

    feature_names = pre.get_feature_names_out()
    importances = rf.feature_importances_

    fi = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    # Cleaner labels for display
    fi["Feature"] = (
        fi["Feature"]
        .str.replace("metal__", "", regex=False)
        .str.replace("remainder__", "", regex=False)
    )
    return fi


# =========================================================
# HELPERS
# =========================================================
def median_for_metal(df, metal, column):
    values = df.loc[df["Heavy metal"] == metal, column]
    if len(values) == 0:
        return float(df[column].median())
    return float(values.median())


def metal_constants(df, metal):
    """Metal properties are constant in this dataset, so use median."""
    cols = [
        "Hydrated radius (nm)",
        "Electronegativity (Pauling)",
        "van der Waals radius (nm)",
        "Molar mass (g/mol)",
    ]
    return {c: median_for_metal(df, metal, c) for c in cols}


def predict_capacity(pipeline, row_dict):
    sample = pd.DataFrame([row_dict], columns=FEATURES)
    return float(pipeline.predict(sample)[0])


def calculate_required_mass(volume_l, concentration_mg_l, capacity_mg_g):
    """
    Theoretical sorbent mass:
        total metal (mg) = volume (L) * concentration (mg/L)
        mass sorbent (g) = total metal (mg) / capacity (mg/g)

    This is a simplified theoretical calculation.
    """
    if capacity_mg_g <= 0:
        return np.nan
    total_metal_mg = volume_l * concentration_mg_l
    return total_metal_mg / capacity_mg_g


# =========================================================
# LOAD
# =========================================================
try:
    df = load_data()
except Exception as e:
    st.error(
        "Не удалось открыть Dataset. S.Lamsiah.xlsx. "
        "Положите Excel-файл в ту же папку, что и app.py."
    )
    st.exception(e)
    st.stop()

pipeline, metrics, test_results = train_model(df)

# =========================================================
# HEADER
# =========================================================
st.title("💧 Интеллектуальная система прогнозирования очистки воды")
st.caption(
    "Machine Learning prototype for predicting heavy-metal adsorption capacity"
)

st.info(
    ""
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Прогноз",
    " Качество модели",
    " Важность признаков",
    " Dataset",
    " О системе"
])

# =========================================================
# TAB 1 — PREDICTION
# =========================================================
with tab1:
    st.subheader("Прогноз сорбционной ёмкости")

    metals = sorted(df["Heavy metal"].dropna().unique().tolist())

    c1, c2 = st.columns(2)

    with c1:
        metal = st.selectbox("Тяжёлый металл", metals)

        activation_temp = st.number_input(
            "Температура активации сорбента, °C",
            value=median_for_metal(df, metal, "Activation temperature (°C)"),
            step=10.0
        )

        bet = st.number_input(
            "BET surface area, m²/g",
            min_value=0.0,
            value=max(0.0, median_for_metal(df, metal, "BET surface area (m²/g)")),
            step=10.0
        )

        pore_diameter = st.number_input(
            "Диаметр пор, nm",
            min_value=0.0,
            value=max(0.0, median_for_metal(df, metal, "Pore diameter (nm)")),
            step=0.1
        )

        pore_volume = st.number_input(
            "Общий объём пор, cm³/g",
            min_value=0.0,
            value=max(0.0, median_for_metal(df, metal, "Total pore volume (cm³/g)\n")),
            step=0.01,
            format="%.3f"
        )

    with c2:
        temperature = st.number_input(
            "Температура эксперимента, °C",
            value=median_for_metal(df, metal, "Temperature (°C)"),
            step=1.0
        )

        ph = st.number_input(
            "pH",
            min_value=0.0,
            max_value=14.0,
            value=float(np.clip(median_for_metal(df, metal, "pH"), 0, 14)),
            step=0.1
        )

        dose = st.number_input(
            "Доза сорбента, g/L",
            min_value=0.001,
            value=max(0.001, median_for_metal(df, metal, "Dose (g/L)")),
            step=0.1
        )

        contact_time = st.number_input(
            "Время контакта, min",
            min_value=0.1,
            value=max(0.1, median_for_metal(df, metal, "Contact time (min)")),
            step=1.0
        )

        initial_concentration = st.number_input(
            "Начальная концентрация, mg/L",
            min_value=0.1,
            value=max(0.1, median_for_metal(df, metal, "Initial concentration (mg/L)")),
            step=1.0
        )

    constants = metal_constants(df, metal)

    with st.expander("Физико-химические свойства выбранного металла"):
        st.write(
            {
                "Hydrated radius (nm)": round(constants["Hydrated radius (nm)"], 4),
                "Electronegativity (Pauling)": round(constants["Electronegativity (Pauling)"], 4),
                "van der Waals radius (nm)": round(constants["van der Waals radius (nm)"], 4),
                "Molar mass (g/mol)": round(constants["Molar mass (g/mol)"], 4),
            }
        )

    row = {
        "Heavy metal": metal,
        "Activation temperature (°C)": activation_temp,
        "Hydrated radius (nm)": constants["Hydrated radius (nm)"],
        "Electronegativity (Pauling)": constants["Electronegativity (Pauling)"],
        "van der Waals radius (nm)": constants["van der Waals radius (nm)"],
        "Molar mass (g/mol)": constants["Molar mass (g/mol)"],
        "BET surface area (m²/g)": bet,
        "Pore diameter (nm)": pore_diameter,
        "Total pore volume (cm³/g)\n": pore_volume,
        "Temperature (°C)": temperature,
        "pH": ph,
        "Dose (g/L)": dose,
        "Contact time (min)": contact_time,
        "Initial concentration (mg/L)": initial_concentration,
    }

    st.markdown("#### Дополнительный расчёт массы сорбента")
    m1, m2 = st.columns(2)
    with m1:
        water_volume = st.number_input(
            "Объём воды, L",
            min_value=0.1,
            value=10.0,
            step=0.5
        )
    with m2:
        use_same_concentration = st.checkbox(
            "Использовать начальную концентрацию выше",
            value=True
        )

    if st.button("Рассчитать прогноз", type="primary"):
        predicted_capacity = predict_capacity(pipeline, row)

        concentration_for_mass = (
            initial_concentration if use_same_concentration
            else st.session_state.get("manual_concentration", initial_concentration)
        )

        required_mass = calculate_required_mass(
            water_volume,
            concentration_for_mass,
            predicted_capacity
        )

        r1, r2, r3 = st.columns(3)
        r1.metric(
            "Прогноз сорбционной ёмкости",
            f"{predicted_capacity:.2f} mg/g"
        )
        r2.metric(
            "Металл в воде",
            f"{water_volume * concentration_for_mass:.2f} mg"
        )
        r3.metric(
            "Теоретическая масса сорбента",
            f"{required_mass:.2f} g"
        )

        st.warning(
            "Расчёт массы сорбента является теоретическим: он предполагает, "
            "что прогнозируемая сорбционная ёмкость полностью реализуется. "
            "Для практического применения требуется лабораторная проверка."
        )


# =========================================================
# TAB 2 — METRICS
# =========================================================
with tab2:
    st.subheader("Оценка модели на тестовой выборке")

    st.write(
        f"Dataset: **{len(df)} наблюдений**. "
        f"Обучение: **{metrics['n_train']}**, тест: **{metrics['n_test']}**."
    )

    a, b, c = st.columns(3)
    a.metric("R²", f"{metrics['R2']:.3f}")
    b.metric("MAE", f"{metrics['MAE']:.2f} mg/g")
    c.metric("RMSE", f"{metrics['RMSE']:.2f} mg/g")

    st.caption(
        "Разделение выполнено случайно в пропорции 80/20 с random_state=42. "
        "Эти показатели относятся только к данной схеме тестирования."
    )

    # Actual vs predicted chart
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        test_results["Actual capacity (mg/g)"],
        test_results["Predicted capacity (mg/g)"],
        alpha=0.7
    )
    min_v = min(
        test_results["Actual capacity (mg/g)"].min(),
        test_results["Predicted capacity (mg/g)"].min()
    )
    max_v = max(
        test_results["Actual capacity (mg/g)"].max(),
        test_results["Predicted capacity (mg/g)"].max()
    )
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    ax.set_xlabel("Actual capacity, mg/g")
    ax.set_ylabel("Predicted capacity, mg/g")
    ax.set_title("Actual vs Predicted")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.markdown("#### Примеры прогнозов")
    st.dataframe(
        test_results[
            [
                "Heavy metal",
                "Contact time (min)",
                "Initial concentration (mg/L)",
                "Actual capacity (mg/g)",
                "Predicted capacity (mg/g)",
                "Absolute error (mg/g)"
            ]
        ].head(30),
        use_container_width=True
    )


# =========================================================
# TAB 3 — FEATURE IMPORTANCE
# =========================================================
with tab3:
    st.subheader("Какие признаки сильнее влияют на прогноз")

    fi = get_feature_importance(pipeline)
    st.dataframe(fi, use_container_width=True)

    top = fi.head(12).sort_values("Importance")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["Feature"], top["Importance"])
    ax.set_xlabel("Feature importance")
    ax.set_title("Random Forest feature importance")
    ax.grid(True, axis="x", alpha=0.3)
    st.pyplot(fig)

    st.caption(
        "Feature importance показывает, насколько часто и насколько полезно "
        "Random Forest использовал признак при построении деревьев. "
        "Это не доказывает причинно-следственную связь."
    )


# =========================================================
# TAB 4 — DATASET
# =========================================================
with tab4:
    st.subheader("Открытый экспериментальный набор данных")

    st.write(f"Количество записей: **{len(df)}**")
    st.write(f"Количество металлов: **{df['Heavy metal'].nunique()}**")

    counts = (
        df["Heavy metal"]
        .value_counts()
        .rename_axis("Heavy metal")
        .reset_index(name="Number of observations")
    )
    st.dataframe(counts, use_container_width=True)

    st.markdown("#### Фрагмент исходных данных")
    st.dataframe(df.head(100), use_container_width=True)

    st.download_button(
        "Скачать подготовленный CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="heavy_metal_adsorption_dataset.csv",
        mime="text/csv"
    )


# =========================================================
# TAB 5 — ABOUT
# =========================================================
with tab5:
    st.subheader("Архитектура системы")

    st.markdown(
        """
**1. Источник данных**  
Открытые экспериментальные данные по адсорбции тяжёлых металлов активированным углём.

**2. Подготовка данных**  
Категориальный признак `Heavy metal` кодируется автоматически.  
Числовые экспериментальные и физико-химические параметры передаются модели без масштабирования.

**3. Модель**  
`RandomForestRegressor` из библиотеки scikit-learn.

**4. Выход модели**  
Прогноз `Adsorption capacity (mg/g)`.

**5. Веб-интерфейс**  
Streamlit позволяет вводить условия эксперимента и получать прогноз без работы непосредственно с Python-кодом.
        """
    )

    st.warning(
        "Ограничение: высокая точность на случайном train/test split не означает, "
        "что модель с такой же точностью будет работать для совершенно нового "
        "сорбента или данных из новой лаборатории. Для этого нужна независимая валидация."
    )
