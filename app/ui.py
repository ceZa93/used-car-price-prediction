import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))

from src.predict import predict_car_price

st.set_page_config(page_title="Procena cene polovnog automobila", layout="centered")

@st.cache_resource
def load_options():
    df = pd.read_csv("data/raw/serbia_car_sales_price_2024_v2.csv")
    if "car_mileage, km" in df.columns:
        df = df.rename(columns={"car_mileage, km": "mileage"})

    # Koristi sve dostupne car_name vrijednosti (npr. "Dacia Duster", "Alfa Romeo 156")
    car_names = (
        df["car_name"]
        .astype("string")
        .dropna()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    car_names = sorted(car_names)

    categorical = ["A/C", "emission_class", "color", "fuel", "car_type", "type_of_drive", "gearbox", "doors"]
    options = {}
    for col in categorical:
        if col in df.columns:
            values = (
                df[col]
                .astype("string")
                .dropna()
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .sort_values()
                .unique()
                .tolist()
            )
            options[col] = values[:100]
        else:
            options[col] = ["Unknown"]

    return options, car_names

options, car_names = load_options()

st.title("Procena cene polovnog automobila")
st.write("Izaberi karakteristike automobila i dobićeš procenu cene.")

col1, col2 = st.columns(2)

with col1:
    car_name = st.selectbox("Model automobila", car_names)
    year = st.number_input("Godina proizvodnje", min_value=1980, max_value=2024, value=2015, step=1)
    mileage = st.number_input("Kilometraža (km)", min_value=0, max_value=1000000, value=150000, step=1000)
    horsepower = st.number_input("Konjske snage (KS)", min_value=50, max_value=1000, value=100, step=5)
    engine = st.number_input("Zapremina motora (cc)", min_value=600, max_value=8000, value=1500, step=100)
    seats = st.number_input("Broj sedišta", min_value=1, max_value=9, value=5, step=1)

with col2:
    ac = st.selectbox("Klima", options["A/C"])
    emission = st.selectbox("Emisiona norma", options["emission_class"])
    color = st.selectbox("Boja", options["color"])
    fuel = st.selectbox("Gorivo", options["fuel"])
    car_type = st.selectbox("Karoserija", options["car_type"])
    drive = st.selectbox("Pogon", options["type_of_drive"])
    gearbox = st.selectbox("Menjač", options["gearbox"])
    doors = st.selectbox("Broj vrata", options["doors"])

if st.button("Proceni cenu", type="primary"):
    car = {
        "car_name": car_name,
        "year": year,
        "mileage": mileage,
        "horsepower": horsepower,
        "engine_capacity, cc": float(engine),
        "seats_amount": float(seats),
        "A/C": ac,
        "emission_class": emission,
        "color": color,
        "fuel": fuel,
        "car_type": car_type,
        "type_of_drive": drive,
        "gearbox": gearbox,
        "doors": doors,
    }
    price = predict_car_price(car)
    st.success(f"Procenjena cena: {price:,.0f} €")
