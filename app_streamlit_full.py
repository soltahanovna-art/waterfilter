
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

from openpyxl.styles import Font, PatternFill, Alignment


# ======================================================
# WATERFILTER AI
# ======================================================

st.set_page_config(
    page_title="WaterFilter AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DATA_FILE = "Dataset. S.Lamsiah.xlsx"
TARGET = "Adsorption capacity (mg/g)"

FEATURES = [
    "Dose (g/L)",
    "Initial concentration (mg/L)",
    "Contact time (min)",
    "pH",
    "BET surface area (m²/g)",
    "Pore diameter (nm)",
    "Total pore volume (cm³/g)",
    "Activation temperature (°C)",
    "van der Waals radius (nm)",
    "Temperature (°C)",
    "Molar mass (g/mol)",
    "Hydrated radius (nm)",
    "Electronegativity (Pauling)"
]

METAL_PROPERTIES = [
    "van der Waals radius (nm)",
    "Molar mass (g/mol)",
    "Hydrated radius (nm)",
    "Electronegativity (Pauling)"
]

USER_FEATURES = [
    feature for feature in FEATURES
    if feature not in METAL_PROPERTIES
]


# ======================================================
# DESIGN
# ======================================================

st.markdown("""
<style>

/* MAIN BACKGROUND */

.stApp {
    background-color: #F4F8FD;
}

.block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* TITLES */

h1, h2, h3 {
    color: #173D67 !important;
    font-family: Arial, sans-serif;
}

/* HEADER */

.app-header {
    background: linear-gradient(
        120deg,
        #163F70,
        #28699E
    );
    padding: 30px 35px;
    border-radius: 15px;
    margin-bottom: 26px;
}

.app-title {
    color: #FFFFFF;
    font-size: 36px;
    font-weight: 750;
    margin-bottom: 6px;
}

.app-subtitle {
    color: #DCEAF8;
    font-size: 16px;
    margin: 0;
}

/* FORM */

div[data-testid="stForm"] {
    background-color: #FFFFFF;
    border: 1px solid #B8D0E8;
    border-radius: 15px;
    padding: 24px;
}

div[data-testid="stWidgetLabel"] p {
    color: #244B73;
    font-weight: 500;
}

/* INPUTS */

div[data-baseweb="input"] {
    border-radius: 8px;
}

div[data-baseweb="select"] {
    border-radius: 8px;
}

/* BUTTON */

div.stFormSubmitButton button[kind="primary"] {
    background-color: #245A91;
    border: 1px solid #245A91;
    color: white;
    border-radius: 9px;
    height: 52px;
    font-size: 17px;
    font-weight: 600;
}

div.stFormSubmitButton button[kind="primary"]:hover {
    background-color: #173D67;
    border-color: #173D67;
    color: white;
}

/* RESULT */

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #B8D0E8;
    border-radius: 12px;
    padding: 22px;
}

div[data-testid="stMetricLabel"] {
    color: #496B8D;
}

div[data-testid="stMetricValue"] {
    color: #174C83;
    font-size: 32px;
    font-weight: 700;
}

/* DOWNLOAD */

div.stDownloadButton button {
    background-color: #FFFFFF;
    color: #245A91;
    border: 1px solid #245A91;
    border-radius: 9px;
    min-height: 50px;
    font-size: 16px;
    font-weight: 600;
}

div.stDownloadButton button:hover {
    color: #173D67;
    border-color: #173D67;
    background-color: #EAF2FA;
}

</style>
""", unsafe_allow_html=True)


# ======================================================
# LOAD DATASET
# ======================================================

def find_dataset():

    folder = Path(__file__).resolve().parent

    preferred = folder / DATA_FILE

    if preferred.exists():
        return preferred

    files = sorted(folder.glob("Dataset*.xlsx"))

    if files:
        return files[0]

    return None


@st.cache_data(show_spinner=False)
def load_data(path, version):

    data = pd.read_excel(path)

    data.columns = (
        data.columns.astype(str).str.strip()
    )

    required = FEATURES + [TARGET, "Heavy metal"]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: " + ", ".join(missing)
        )

    for col in FEATURES + [TARGET]:
        data[col] = pd.to_numeric(
            data[col],
            errors="raise"
        )

    values = data[FEATURES + [TARGET]].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Dataset contains invalid numerical values."
        )

    if len(data) != 1528:
        raise ValueError(
            f"Expected 1528 records, found {len(data)}."
        )

    return data


# ======================================================
# GRADIENT BOOSTING MODEL
# ======================================================

@st.cache_resource(
    show_spinner="Подготовка модели Gradient Boosting..."
)
def train_model(path, version):

    data = load_data(path, version)

    regressor = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.10,
        max_depth=15,
        subsample=0.7,
        max_features=0.7,
        min_samples_split=10,
        min_samples_leaf=6,
        loss="squared_error",
        random_state=42,
        validation_fraction=0.15,
        n_iter_no_change=20,
        tol=0.001
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", regressor)
    ])

    pipeline.fit(
        data[FEATURES],
        data[TARGET]
    )

    return pipeline


dataset_path = find_dataset()

if dataset_path is None:

    st.error(
        "Не найден файл Dataset. S.Lamsiah.xlsx"
    )
    st.stop()

try:

    version = dataset_path.stat().st_mtime_ns

    df = load_data(
        str(dataset_path),
        version
    )

    model = train_model(
        str(dataset_path),
        version
    )

except Exception:

    st.error(
        "Ошибка загрузки данных или модели."
    )
    st.stop()


# ======================================================
# DEFAULT VALUES
# ======================================================

def get_default_row(metal):

    metal_data = df.loc[
        df["Heavy metal"] == metal
    ].copy()

    # Для каждого параметра определяем медиану
    medians = metal_data[USER_FEATURES].median()

    # Масштабируем отклонения для сравнения
    spread = (
        metal_data[USER_FEATURES].quantile(0.75)
        - metal_data[USER_FEATURES].quantile(0.25)
    )

    spread = spread.replace(0, 1.0)

    # Ищем одно наиболее типичное наблюдение
    distances = (
        metal_data[USER_FEATURES]
        .sub(medians)
        .abs()
        .div(spread)
        .sum(axis=1)
    )

    closest_index = distances.idxmin()

    # Возвращаем всю строку одного эксперимента
    return metal_data.loc[closest_index]


def get_metal_properties(metal):

    subset = df.loc[
        df["Heavy metal"] == metal
    ]

    return {
        feature: float(subset[feature].median())
        for feature in METAL_PROPERTIES
    }


def input_field(
    label,
    feature,
    key,
    default_row,
    metal,
    minimum,
    step,
    fmt="%.2f",
    maximum=None
):

    default_value = max(
        minimum,
        float(default_row[feature])
    )

    if maximum is not None:
        default_value = min(
            default_value,
            maximum
        )

    return st.number_input(
        label,
        min_value=float(minimum),
        max_value=(
            float(maximum)
            if maximum is not None
            else None
        ),
        value=float(default_value),
        step=float(step),
        format=fmt,
        key=f"{metal}_{key}"
    )


# ======================================================
# EXCEL EXPORT
# ======================================================

def create_excel(record):

    output = BytesIO()

    data = pd.DataFrame([record])

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        data.to_excel(
            writer,
            index=False,
            sheet_name="Результат"
        )

        sheet = writer.sheets["Результат"]

        # Header formatting
        blue_fill = PatternFill(
            start_color="245A91",
            end_color="245A91",
            fill_type="solid"
        )

        white_font = Font(
            color="FFFFFF",
            bold=True
        )

        for cell in sheet[1]:
            cell.fill = blue_fill
            cell.font = white_font
            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True
            )

        sheet.row_dimensions[1].height = 38
        sheet.freeze_panes = "A2"

        for column in sheet.columns:

            letter = column[0].column_letter

            width = max(
                len(str(cell.value or ""))
                for cell in column
            )

            sheet.column_dimensions[letter].width = min(
                max(width + 3, 16),
                40
            )

    return output.getvalue()


# ======================================================
# HEADER
# ======================================================

st.markdown("""
<div class="app-header">
    <div class="app-title">WaterFilter AI</div>
    <p class="app-subtitle">
        Прогнозирование сорбционной ёмкости
        активированного угля
    </p>
</div>
""", unsafe_allow_html=True)


# ======================================================
# METAL SELECTION
# ======================================================

metals = sorted(
    df["Heavy metal"].dropna().unique().tolist()
)

metal = st.selectbox(
    "Загрязнитель",
    metals,
    key="selected_metal"
)

default_row = get_default_row(metal)


# ======================================================
# INPUT FORM
# ======================================================

with st.form("prediction_form"):

    left, right = st.columns(
        2,
        gap="large"
    )

    # ----------------------------------
    # LEFT COLUMN — WATER
    # ----------------------------------

    with left:

        st.subheader("Параметры воды")

        concentration = input_field(
            "Концентрация металла, мг/л",
            "Initial concentration (mg/L)",
            "concentration",
            default_row,
            metal,
            0.001,
            1.0,
            "%.3f"
        )

        water_volume = st.number_input(
            "Объём воды, л",
            min_value=0.01,
            value=10.0,
            step=1.0,
            format="%.2f"
        )

        ph = input_field(
            "pH воды",
            "pH",
            "ph",
            default_row,
            metal,
            0.0,
            0.1,
            "%.2f",
            14.0
        )

        temperature = input_field(
            "Температура воды, °C",
            "Temperature (°C)",
            "temperature",
            default_row,
            metal,
            0.0,
            1.0
        )

        contact_time = input_field(
            "Время контакта, мин",
            "Contact time (min)",
            "contact_time",
            default_row,
            metal,
            0.1,
            1.0
        )

        target_removal = st.number_input(
            "Планируемая доля удаления, %",
            min_value=1,
            max_value=100,
            value=90,
            step=1
        )

    # ----------------------------------
    # RIGHT COLUMN — SORBENT
    # ----------------------------------

    with right:

        st.subheader(
            "Характеристики активированного угля"
        )

        dose = input_field(
            "Дозировка угля, г/л",
            "Dose (g/L)",
            "dose",
            default_row,
            metal,
            0.001,
            0.1,
            "%.3f"
        )

        bet = input_field(
            "Площадь поверхности BET, м²/г",
            "BET surface area (m²/g)",
            "bet",
            default_row,
            metal,
            0.001,
            10.0
        )

        pore_diameter = input_field(
            "Диаметр пор, нм",
            "Pore diameter (nm)",
            "pore_diameter",
            default_row,
            metal,
            0.001,
            0.1,
            "%.3f"
        )

        pore_volume = input_field(
            "Общий объём пор, см³/г",
            "Total pore volume (cm³/g)",
            "pore_volume",
            default_row,
            metal,
            0.0001,
            0.01,
            "%.4f"
        )

        activation = input_field(
            "Температура активации, °C",
            "Activation temperature (°C)",
            "activation",
            default_row,
            metal,
            0.0,
            10.0
        )

    st.markdown("")

    submitted = st.form_submit_button(
        "Рассчитать прогноз",
        type="primary",
        use_container_width=True
    )


# ======================================================
# PREDICTION
# ======================================================

if submitted:

    st.session_state.pop(
        "prediction_result",
        None
    )

    properties = get_metal_properties(metal)

    values = {
        "Dose (g/L)": dose,
        "Initial concentration (mg/L)": concentration,
        "Contact time (min)": contact_time,
        "pH": ph,
        "BET surface area (m²/g)": bet,
        "Pore diameter (nm)": pore_diameter,
        "Total pore volume (cm³/g)": pore_volume,
        "Activation temperature (°C)": activation,
        "Temperature (°C)": temperature
    }

    values.update(properties)

    sample = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    predicted_qe = float(
        model.predict(sample)[0]
    )

    # Проверка баланса массы
    max_possible_qe = concentration / dose

    if (
        not np.isfinite(predicted_qe)
        or predicted_qe <= 0
        or predicted_qe > max_possible_qe + 1e-8
    ):

        st.error(
            "Для этих параметров не удалось "
            "получить допустимый прогноз. "
            "Измените входные значения."
        )

    else:

        # Metal mass to be removed (mg)
        metal_to_remove = (
            concentration
            * water_volume
            * target_removal
            / 100
        )

        # Theoretical sorbent mass (g)
        required_mass = (
            metal_to_remove / predicted_qe
        )

        export_record = {
            "Загрязнитель": metal,
            "Концентрация, мг/л": concentration,
            "Объём воды, л": water_volume,
            "pH": ph,
            "Температура воды, °C": temperature,
            "Время контакта, мин": contact_time,
            "Желаемое удаление, %": target_removal,
            "Дозировка угля, г/л": dose,
            "BET, м²/г": bet,
            "Диаметр пор, нм": pore_diameter,
            "Объём пор, см³/г": pore_volume,
            "Температура активации, °C": activation,
            "Прогноз Qe, мг/г": round(
                predicted_qe, 4
            ),
            "Теоретическая масса угля, г": round(
                required_mass, 4
            )
        }

        st.session_state["prediction_result"] = {
            "metal": metal,
            "qe": predicted_qe,
            "mass": required_mass,
            "excel": create_excel(export_record)
        }


# ======================================================
# RESULTS
# ======================================================

result = st.session_state.get(
    "prediction_result"
)

if result is not None and result["metal"] == metal:

    st.divider()

    st.subheader("Результат расчёта")

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    with col1:

        st.metric(
            "Сорбционная ёмкость Qe",
            f"{result['qe']:.2f} мг/г"
        )

    with col2:

        st.metric(
            "Теоретическая масса угля",
            f"{result['mass']:.2f} г"
        )

    st.markdown("")

    st.download_button(
        "Скачать результат (Excel)",
        data=result["excel"],
        file_name="WaterFilterAI_Result.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True
    )

    st.caption(
        "Расчёт является ориентировочным "
        "и требует экспериментального подтверждения."
    )
