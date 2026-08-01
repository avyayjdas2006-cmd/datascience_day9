"""
generate_synthetic_data.py
---------------------------------
Generates a synthetic dataset that mirrors the SCHEMA and general price
relationships of the CarDekho Used Car Price dataset:
https://www.kaggle.com/datasets/manishkr1754/cardekho-used-car-data

WHY THIS EXISTS
This sandbox cannot reach kaggle.com to download the real CSV. This script
lets the project run end-to-end (train + app) out of the box. For your real
submission, download the actual Kaggle CSV, save it as
`data/cardekho_dataset.csv` with the same column names, and re-run
`train_model.py` — the app code does not need to change.

Columns produced (same as the real dataset):
    brand, model, vehicle_age, km_driven, seller_type, fuel_type,
    transmission_type, mileage, engine, max_power, seats, selling_price
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 6000

BRANDS = {
    # brand: (models, base_price_lakhs, premium_factor)
    "Maruti":    (["Swift", "Baleno", "Alto", "Dzire", "Wagon R"], 5.5, 1.0),
    "Hyundai":   (["i20", "Creta", "Venue", "i10", "Verna"], 7.0, 1.05),
    "Honda":     (["City", "Amaze", "Jazz", "WR-V"], 7.5, 1.1),
    "Toyota":    (["Innova", "Fortuner", "Glanza", "Yaris"], 10.0, 1.25),
    "Tata":      (["Nexon", "Punch", "Altroz", "Harrier"], 7.0, 1.0),
    "Mahindra":  (["XUV500", "Scorpio", "Bolero", "XUV300"], 9.0, 1.1),
    "Ford":      (["EcoSport", "Figo", "Endeavour"], 7.5, 0.95),
    "BMW":       (["3 Series", "5 Series", "X1"], 35.0, 2.2),
    "Audi":      (["A4", "A6", "Q3"], 38.0, 2.1),
    "Mercedes-Benz": (["C-Class", "E-Class", "GLA"], 42.0, 2.3),
    "Kia":       (["Seltos", "Sonet", "Carens"], 8.5, 1.1),
    "Renault":   (["Kwid", "Duster", "Triber"], 5.0, 0.9),
    "Skoda":     (["Rapid", "Octavia", "Kushaq"], 9.5, 1.15),
    "Volkswagen":(["Polo", "Vento", "Taigun"], 8.5, 1.05),
}

SELLER_TYPES = ["Individual", "Dealer", "Trustmark Dealer"]
FUEL_TYPES = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
FUEL_WEIGHTS = [0.52, 0.36, 0.06, 0.02, 0.04]
TRANSMISSIONS = ["Manual", "Automatic"]

rows = []
brand_list = list(BRANDS.keys())

for _ in range(N):
    brand = RNG.choice(brand_list)
    models, base_price, premium = BRANDS[brand]
    model = RNG.choice(models)

    vehicle_age = int(RNG.integers(0, 18))  # years
    km_driven = int(max(500, RNG.normal(15000, 8000) * (vehicle_age + 1) / 3))
    seller_type = RNG.choice(SELLER_TYPES, p=[0.55, 0.35, 0.10])
    fuel_type = RNG.choice(FUEL_TYPES, p=FUEL_WEIGHTS)
    transmission_type = RNG.choice(TRANSMISSIONS, p=[0.72, 0.28])

    mileage = round(float(np.clip(RNG.normal(18 if fuel_type != "Electric" else 0, 4), 8, 32)), 2)
    engine = int(np.clip(RNG.normal(1350 * premium, 250), 700, 3500))
    max_power = round(float(np.clip(RNG.normal(90 * premium, 25), 35, 400)), 2)
    seats = int(RNG.choice([4, 5, 5, 5, 6, 7], p=[0.05, 0.55, 0.0, 0.0, 0.15, 0.25]))
    # (weights sum fixed below)
    seats = int(RNG.choice([5, 7, 5, 4, 6]))

    # --- price model (in INR Lakhs -> converted to rupees) ---
    price_lakhs = base_price * premium
    price_lakhs *= (0.93 ** vehicle_age)                 # depreciation with age
    price_lakhs *= max(0.55, 1 - km_driven / 300000)      # depreciation with km
    price_lakhs *= 1.15 if transmission_type == "Automatic" else 1.0
    price_lakhs *= 1.10 if seller_type == "Trustmark Dealer" else (1.0 if seller_type == "Dealer" else 0.95)
    price_lakhs *= (max_power / 90) ** 0.35
    price_lakhs *= 1.05 if fuel_type == "Diesel" else (0.9 if fuel_type == "CNG" else 1.0)
    price_lakhs *= float(RNG.normal(1.0, 0.06))           # market noise
    price_lakhs = max(0.8, price_lakhs)

    selling_price = int(price_lakhs * 100000)

    rows.append({
        "brand": brand,
        "model": model,
        "vehicle_age": vehicle_age,
        "km_driven": km_driven,
        "seller_type": seller_type,
        "fuel_type": fuel_type,
        "transmission_type": transmission_type,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        "selling_price": selling_price,
    })

df = pd.DataFrame(rows)
out_path = "data/cardekho_dataset.csv"
df.to_csv(out_path, index=False)
print(f"Wrote {len(df)} rows to {out_path}")
print(df.head())
