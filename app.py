import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Загрузка данных и модели
@st.cache_data
def load_reference_data():
    return pd.read_csv('data/df_train_processed.csv')

@st.cache_resource
def load_model_artifacts():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

# Подготовка данных
try:
    ref_df = load_reference_data()
    artifacts = load_model_artifacts()
except FileNotFoundError as e:
    st.error(f"Не найден файл: {e}")
    st.stop()

model = artifacts['model']
scaler = artifacts['scaler']
target_encoder = artifacts['target_encoder']
one_hot_enc = artifacts['one_hot_encoder']
num_features = artifacts['num_features']
cat_features = artifacts['cat_features']
text_features = artifacts['text_features']
final_feature_names = artifacts['final_feature_names']

def preprocess_input(df_raw):
    df = df_raw.copy()
    df[text_features] = target_encoder.transform(df[text_features])
    df[num_features] = scaler.transform(df[num_features])
    cat_enc = one_hot_enc.transform(df[cat_features])
    cat_df = pd.DataFrame(
        cat_enc,
        columns=one_hot_enc.get_feature_names_out(cat_features),
        index=df.index
    )
    non_cat = num_features + text_features
    processed = pd.concat([df[non_cat], cat_df], axis=1)
    return processed[final_feature_names].values

# Streamlit UI
st.set_page_config(page_title="Прогноз цены автомобиля", layout="wide")
st.title("Прогноз цены подержанного автомобиля")
st.markdown("Модель: Ridge Regression (R² ≈ 0.897 на тесте)")

#Выбор данных для EDA
st.sidebar.header("Источник данных для EDA")
eda_option = st.sidebar.radio(
    "Откуда брать данные для графиков?",
    ("Исходный трейн (рекомендуется)", "Загрузить свой CSV")
)

if eda_option == "Загрузить свой CSV":
    uploaded_eda = st.sidebar.file_uploader("Загрузите CSV для EDA", type=["csv"])
    if uploaded_eda:
        df_eda = pd.read_csv(uploaded_eda)
        st.sidebar.success("CSV загружен")
    else:
        df_eda = ref_df
else:
    df_eda = ref_df

# EDA блок
st.header("EDA")

# Распределение целевой переменной
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(df_eda['selling_price'], bins=40, color='#4e79a7', edgecolor='black')
ax[0].set_title('Распределение цены')
ax[1].hist(df_eda['log_selling_price'], bins=40, color="#33f10c", edgecolor='black')
ax[1].set_title('Распределение log(цены)')
st.pyplot(fig)

# Год vs цена
st.subheader("Год выпуска и цена")
fig, ax = plt.subplots(figsize=(10, 3))
sns.scatterplot(data=df_eda, x='year', y='selling_price', alpha=0.6, ax=ax)
ax.set_ylabel('Цена')
st.pyplot(fig)

# Распределение топлива
st.subheader("Распределение по типу топлива")
fig, ax = plt.subplots(figsize=(8, 3))
df_eda['fuel'].value_counts().plot(kind='bar', color='#59a14f', ax=ax)
ax.set_ylabel('Количество')
plt.xticks(rotation=0)
st.pyplot(fig)

# Boxplot: топливо vs цена
st.subheader("Средняя цена по типу топлива")
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=df_eda, x='fuel', y='selling_price', errorbar=('ci', 95), ax=ax)
ax.set_ylabel('Средняя цена')
plt.xticks(rotation=15)
st.pyplot(fig)

# Пробег
st.subheader("Распределение пробега")
fig, ax = plt.subplots(figsize=(10, 3))
sns.histplot(df_eda['km_driven'], bins=50, color='#e15759', kde=True, ax=ax)
ax.set_xlabel('Пробег (км)')
st.pyplot(fig)

# Корреляции
st.subheader("Матрица корреляций (численные признаки)")
num_cols = ['year', 'km_driven', 'mileage_kmpl', 'engine_cc', 'max_power_bhp', 'torque_nm', 'log_selling_price']
corr = df_eda[num_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
st.pyplot(fig)

# Ввод данных
st.header("Введите данные автомобиля")

input_method = st.radio("Способ ввода:", ("Ручной ввод", "Загрузить CSV"))

if input_method == "Ручной ввод":
    with st.form("manual_input"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Модель (name)", "Maruti Swift Dzire VDI")
            year = st.number_input("Год (year)", 1980, 2025, 2015)
            km_driven = st.number_input("Пробег (km_driven)", 0, 3000000, 50000)
            mileage = st.number_input("Расход (mileage_kmpl)", 0.0, 50.0, 20.0, step=1.)
            engine = st.number_input("Двигатель (engine_cc)", 500, 5000, 1200)
            max_power = st.number_input("Мощность (max_power_bhp)", 0.0, 500.0, 80.0, step=1.)
            torque = st.number_input("Крутящий момент (torque_nm)", 0.0, 1000.0, 150.0, step=1.)
        with col2:
            fuel = st.selectbox("Топливо", ["Petrol", "Diesel", "CNG", "LPG"])
            seller = st.selectbox("Продавец", ["Individual", "Dealer", "Trustmark Dealer"])
            trans = st.selectbox("КПП", ["Manual", "Automatic"])
            owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
            seats = st.selectbox("Места", [2, 3, 4, 5, 6, 7, 8, 9])
        submitted = st.form_submit_button("Предсказать")
    
    if submitted:
        input_df = pd.DataFrame([{
            'name': name,
            'year': year,
            'km_driven': km_driven,
            'mileage_kmpl': mileage,
            'engine_cc': engine,
            'max_power_bhp': max_power,
            'torque_nm': torque,
            'fuel': fuel,
            'seller_type': seller,
            'transmission': trans,
            'owner': owner,
            'seats': seats
        }])
        X = preprocess_input(input_df)
        log_pred = model.predict(X)[0]
        price = np.exp(log_pred)
        st.success(f"### Предсказанная цена: **{price:,.0f}**")

else:
    st.info("Загрузите CSV с колонками: name, year, km_driven, mileage_kmpl, engine_cc, max_power_bhp, torque_nm, fuel, seller_type, transmission, owner, seats")
    uploaded_csv = st.file_uploader("Выберите CSV", type=["csv"])
    if uploaded_csv:
        input_df = pd.read_csv(uploaded_csv)
        st.write("Загруженные данные:")
        st.dataframe(input_df.head())
        try:
            X = preprocess_input(input_df)
            log_preds = model.predict(X)
            input_df['pred_price'] = np.exp(log_preds)
            st.subheader("Результаты предсказания")
            st.dataframe(input_df[['pred_price']])
            csv_out = input_df.to_csv(index=False).encode('utf-8')
            st.download_button("Скачать результаты", csv_out, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Ошибка при обработке: {e}")

#Веса модели
st.header("Коэффициенты модели Ridge")

coef_df = pd.DataFrame({
    'Признак': final_feature_names,
    'Коэффициент': model.coef_
}).sort_values('Коэффициент', key=abs, ascending=False)

top15 = coef_df.head(15)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['green' if c > 0 else 'red' for c in top15['Коэффициент']]
ax.barh(top15['Признак'], top15['Коэффициент'], color=colors)
ax.set_xlabel('Значение коэффициента')
ax.set_title('Топ-15 признаков по |коэффициент| (зелёный ↑, красный ↓)')
st.pyplot(fig)

st.write("### Таблица коэффициентов")
st.dataframe(
    top15.style.applymap(
        lambda x: 'color: green; font-weight: bold' if x > 0 else 'color: red; font-weight: bold',
        subset=['Коэффициент']
    ),
    use_container_width=True
)