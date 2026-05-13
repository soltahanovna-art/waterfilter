
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

# Опционально: анализ фото
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False


# -------------------------------------------------
# 1. ДАННЫЕ И МОДЕЛЬ
# -------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    data = [
        # metal, time_h, resin_type, capacity_mg_g

        # Cobalt (Co)
        ["Co", 0.5, 0, 11.52],
        ["Co", 1.0, 0, 23.56],
        ["Co", 1.5, 0, 36.34],
        ["Co", 2.0, 0, 38.32],

        ["Co", 0.5, 1, 25.43],
        ["Co", 1.0, 1, 41.93],
        ["Co", 1.5, 1, 62.87],
        ["Co", 2.0, 1, 64.56],

        # Copper (Cu)
        ["Cu", 0.5, 0, 12.63],
        ["Cu", 1.0, 0, 25.24],
        ["Cu", 1.5, 0, 39.69],
        ["Cu", 2.0, 0, 41.49],

        ["Cu", 0.5, 1, 27.04],
        ["Cu", 1.0, 1, 45.10],
        ["Cu", 1.5, 1, 68.10],
        ["Cu", 2.0, 1, 69.53],
    ]
    df = pd.DataFrame(data, columns=["metal", "time_h", "resin_type", "capacity_mg_g"])
    df["metal_code"] = df["metal"].map({"Cu": 0, "Co": 1})
    df["resin_name"] = df["resin_type"].map({
        0: "Без угля",
        1: "С активированным углём"
    })
    return df


@st.cache_resource
def train_model():
    df = load_data()
    X = df[["metal_code", "time_h", "resin_type"]]
    y = df["capacity_mg_g"]

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=6,
        random_state=42
    )
    model.fit(X, y)
    return model


def predict_capacity(model, metal: str, time_h: float, resin_type: int) -> float:
    metal_code = 0 if metal == "Cu" else 1
    sample = pd.DataFrame([{
        "metal_code": metal_code,
        "time_h": time_h,
        "resin_type": resin_type
    }])
    return float(model.predict(sample)[0])


def calculate_required_mass(volume_l: float, concentration_mg_l: float, capacity_mg_g: float) -> float:
    """m = (C * V) / q"""
    total_metal_mg = volume_l * concentration_mg_l
    return total_metal_mg / capacity_mg_g


# -------------------------------------------------
# 2. АНАЛИЗ ФОТО (ПРОТОТИП)
# -------------------------------------------------
def estimate_pollution_from_uploaded_image(uploaded_file):
    if not CV2_AVAILABLE:
        return None, "OpenCV не установлен. Для анализа фото установите opencv-python."

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None, "Не удалось прочитать изображение."

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    mean_r = float(np.mean(image_rgb[:, :, 0]))
    mean_g = float(np.mean(image_rgb[:, :, 1]))
    mean_b = float(np.mean(image_rgb[:, :, 2]))

    # Упрощённая логика для прототипа
    if contrast < 30:
        pollution_level = "Высокий"
        estimated_concentration = 20.0
    elif contrast < 50:
        pollution_level = "Средний"
        estimated_concentration = 10.0
    else:
        pollution_level = "Низкий"
        estimated_concentration = 3.0

    result = {
        "image_rgb": image_rgb,
        "brightness": brightness,
        "contrast": contrast,
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "pollution_level": pollution_level,
        "estimated_concentration_mg_l": estimated_concentration
    }
    return result, None


# -------------------------------------------------
# 3. ГРАФИКИ
# -------------------------------------------------
def plot_experimental_curves(df: pd.DataFrame, metal: str):
    fig, ax = plt.subplots(figsize=(8, 5))

    subset = df[df["metal"] == metal].copy()

    for resin_type, resin_label in [(0, "Без угля"), (1, "С активированным углём")]:
        temp = subset[subset["resin_type"] == resin_type].sort_values("time_h")
        ax.plot(temp["time_h"], temp["capacity_mg_g"], marker='o', label=resin_label)

    ax.set_title(f"Экспериментальная сорбция для {metal}")
    ax.set_xlabel("Время контакта, ч")
    ax.set_ylabel("Сорбционная ёмкость, мг/г")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_predicted_curve(model, metal: str, resin_type: int):
    fig, ax = plt.subplots(figsize=(8, 5))
    time_grid = np.linspace(0.5, 2.5, 60)

    preds = [
        predict_capacity(model, metal, float(t), resin_type)
        for t in time_grid
    ]

    label = "С активированным углём" if resin_type == 1 else "Без угля"
    ax.plot(time_grid, preds, label=f"Прогноз: {label}")
    ax.set_title(f"Прогнозируемая кривая сорбции ({metal})")
    ax.set_xlabel("Время контакта, ч")
    ax.set_ylabel("Сорбционная ёмкость, мг/г")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


# -------------------------------------------------
# 4. ИНТЕРФЕЙС STREAMLIT
# -------------------------------------------------
st.set_page_config(
    page_title="AI-система очистки воды",
    page_icon="💧",
    layout="wide"
)

st.title("💧 AI-система очистки воды с использованием биоразлагаемого сорбента")
st.write(
    "Прототип системы для прогнозирования сорбционной ёмкости и расчёта необходимой массы сорбента "
    "на основе экспериментальных данных по Cu(II) и Co(II)."
)

df = load_data()
model = train_model()

tab1, tab2, tab3, tab4 = st.tabs([
    "Расчёт очистки",
    "Анализ фото воды",
    "Графики",
    "Исходные данные"
])

with tab1:
    st.subheader("Расчёт необходимой массы сорбента")

    col1, col2 = st.columns(2)

    with col1:
        metal = st.selectbox("Металл", ["Cu", "Co"])
        volume_l = st.number_input("Объём воды, л", min_value=0.1, value=20.0, step=0.1)
        concentration_mg_l = st.number_input("Концентрация загрязнения, мг/л", min_value=0.1, value=10.0, step=0.1)

    with col2:
        time_h = st.slider("Время контакта, ч", min_value=0.5, max_value=3.0, value=2.0, step=0.1)
        resin_name = st.selectbox("Тип сорбента", ["Без угля", "С активированным углём"])
        resin_type = 0 if resin_name == "Без угля" else 1

    if st.button("Рассчитать", type="primary"):
        capacity = predict_capacity(model, metal, time_h, resin_type)
        required_mass = calculate_required_mass(volume_l, concentration_mg_l, capacity)

        c1, c2, c3 = st.columns(3)
        c1.metric("Прогнозируемая сорбционная ёмкость", f"{capacity:.2f} мг/г")
        c2.metric("Общий металл в воде", f"{volume_l * concentration_mg_l:.2f} мг")
        c3.metric("Необходимая масса сорбента", f"{required_mass:.2f} г")

        st.info(
            f"При объёме **{volume_l:.1f} л**, концентрации **{concentration_mg_l:.1f} мг/л**, "
            f"металле **{metal}** и времени **{time_h:.1f} ч** "
            f"система прогнозирует ёмкость **{capacity:.2f} мг/г**."
        )

        if resin_type == 1:
            st.success("Модифицированный сорбент с активированным углём обычно показывает более высокую эффективность.")
        else:
            st.warning("Сорбент без угля работает, но обычно уступает модифицированному варианту.")

        # Таблица результата
        result_df = pd.DataFrame([{
            "Металл": metal,
            "Объём воды, л": volume_l,
            "Концентрация, мг/л": concentration_mg_l,
            "Время, ч": time_h,
            "Тип сорбента": resin_name,
            "Сорбционная ёмкость, мг/г": round(capacity, 2),
            "Масса сорбента, г": round(required_mass, 2)
        }])

        st.download_button(
            label="Скачать результат CSV",
            data=result_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="water_purification_result.csv",
            mime="text/csv"
        )

with tab2:
    st.subheader("Прототип оценки загрязнения по фото")
    st.caption("Это демонстрационный модуль. Он не заменяет лабораторный анализ и даёт ориентировочную оценку.")

    uploaded_file = st.file_uploader("Загрузите фото воды", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        result, error = estimate_pollution_from_uploaded_image(uploaded_file)

        if error:
            st.error(error)
        else:
            st.image(result["image_rgb"], caption="Загруженное изображение", use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Яркость", f"{result['brightness']:.2f}")
            c2.metric("Контраст", f"{result['contrast']:.2f}")
            c3.metric("Оценка загрязнения", result["pollution_level"])

            st.write(
                f"Ориентировочная концентрация загрязнения: **{result['estimated_concentration_mg_l']:.1f} мг/л**"
            )

            st.write("Средние значения RGB:")
            st.write(
                f"R = {result['mean_r']:.1f}, "
                f"G = {result['mean_g']:.1f}, "
                f"B = {result['mean_b']:.1f}"
            )

            st.info(
                "Эту оценку можно подставить во вкладке «Расчёт очистки» как примерную концентрацию загрязнения."
            )

with tab3:
    st.subheader("Графики и визуализация")

    left, right = st.columns(2)

    with left:
        metal_for_plot = st.selectbox("Выберите металл для графика", ["Cu", "Co"], key="plot_metal_1")
        fig1 = plot_experimental_curves(df, metal_for_plot)
        st.pyplot(fig1)

    with right:
        metal_for_pred = st.selectbox("Выберите металл для прогноза", ["Cu", "Co"], key="plot_metal_2")
        resin_for_pred = st.selectbox("Выберите сорбент", ["Без угля", "С активированным углём"], key="plot_resin_2")
        resin_for_pred_code = 0 if resin_for_pred == "Без угля" else 1
        fig2 = plot_predicted_curve(model, metal_for_pred, resin_for_pred_code)
        st.pyplot(fig2)

    st.markdown("### Важность признаков")
    importances = model.feature_importances_
    feature_names = ["Тип металла", "Время контакта", "Тип сорбента"]

    fi_df = pd.DataFrame({
        "Признак": feature_names,
        "Важность": importances
    }).sort_values("Важность", ascending=False)

    st.dataframe(fi_df, use_container_width=True)

    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.bar(fi_df["Признак"], fi_df["Важность"])
    ax3.set_title("Важность признаков в модели")
    ax3.set_ylabel("Важность")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

with tab4:
    st.subheader("Исходные экспериментальные данные")
    st.dataframe(df[["metal", "time_h", "resin_name", "capacity_mg_g"]], use_container_width=True)

    st.markdown("### Что принимает система на вход")
    st.markdown(
        """
        - тип металла (**Cu** или **Co**)
        - объём воды, л
        - концентрация загрязнения, мг/л
        - время контакта, ч
        - тип сорбента (**без угля** / **с активированным углём**)
        """
    )

    st.markdown("### Что система выдаёт на выход")
    st.markdown(
        """
        - прогнозируемую сорбционную ёмкость, мг/г
        - общий объём загрязнения в воде, мг
        - необходимую массу сорбента, г
        """
    )

st.markdown("---")
st.caption(
    "Важно: модель обучена на небольшом наборе экспериментальных данных и предназначена "
    "для демонстрации концепции интеллектуальной системы очистки воды."
)
