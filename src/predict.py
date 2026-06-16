import hashlib
import joblib
import pandas as pd
from pathlib import Path

REFERENCE_YEAR = 2024

MODEL_DIR = Path("models")
MODEL_FILES = {
    "model": MODEL_DIR / "best_model.joblib",
    "scaler": MODEL_DIR / "scaler.joblib",
    "encoders": MODEL_DIR / "label_encoders.joblib",
    "brand_price": MODEL_DIR / "brand_price_map.joblib",
}

NUMERIC_RANGES = {
    "year": (1950, REFERENCE_YEAR + 1),
    "mileage": (0, 2_000_000),
    "horsepower": (1, 2000),
    "engine_capacity, cc": (0, 20_000),
    "seats_amount": (1, 12),
}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_model_checksums(out_path: Path = MODEL_DIR / "checksums.sha256"):
    lines = []
    for label, fpath in MODEL_FILES.items():
        if fpath.exists():
            lines.append(f"{_file_sha256(fpath)}  {fpath.name}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"✓ Checksums written to {out_path}")


def verify_model_checksums(checksum_path: Path = MODEL_DIR / "checksums.sha256"):
    if not checksum_path.exists():
        print(
            "Warning: no checksums.sha256 found — skipping integrity check. "
            "Run generate_model_checksums() after training to enable verification."
        )
        return
    expected = {}
    for line in checksum_path.read_text().splitlines():
        if line.strip():
            digest, name = line.split("  ", 1)
            expected[name] = digest
    for label, fpath in MODEL_FILES.items():
        if fpath.name in expected:
            actual = _file_sha256(fpath)
            if actual != expected[fpath.name]:
                raise RuntimeError(
                    f"Integrity check failed for {fpath.name}: "
                    f"expected {expected[fpath.name]}, got {actual}. "
                    "The model file may have been tampered with."
                )


def _validate_input(data: dict) -> dict:
    validated = {}
    for key, value in data.items():
        if value is None:
            raise ValueError(f"Field '{key}' must not be None.")

        if key in NUMERIC_RANGES:
            try:
                num = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Field '{key}' must be numeric, got {type(value).__name__}: {value!r}"
                )
            lo, hi = NUMERIC_RANGES[key]
            if not (lo <= num <= hi):
                raise ValueError(
                    f"Field '{key}' value {num} is outside the allowed range [{lo}, {hi}]."
                )
            validated[key] = num
        else:
            validated[key] = value
    return validated


def predict_car_price(input_data):
    verify_model_checksums()

    model = joblib.load(MODEL_FILES["model"])
    scaler = joblib.load(MODEL_FILES["scaler"])
    encoders = joblib.load(MODEL_FILES["encoders"])
    brand_price = joblib.load(MODEL_FILES["brand_price"])

    input_data = _validate_input(dict(input_data))

    if 'year' in input_data and 'car_age' not in input_data:
        input_data['car_age'] = REFERENCE_YEAR - int(input_data['year'])
    input_data.pop('year', None)

    if 'km_per_year' not in input_data and 'mileage' in input_data and 'car_age' in input_data:
        age = input_data['car_age'] if input_data['car_age'] != 0 else 1
        input_data['km_per_year'] = input_data['mileage'] / age

    if 'brand' in input_data:
        brand_key = str(input_data['brand']).upper()
        input_data['brand'] = brand_price['map'].get(brand_key, brand_price['global_mean'])
        if brand_key not in brand_price['map']:
            print(f"Upozorenje: brend '{brand_key}' nije viđen u treningu, koristi se prosečna cena.")

    for col, value in input_data.items():
        if col in encoders:
            val_str = str(value)

            allowed_values = encoders[col].classes_

            if val_str in allowed_values:
                input_data[col] = encoders[col].transform([val_str])[0]
            else:
                print(f"Upozorenje: Vrednost '{val_str}' za '{col}' nije nađena. Dostupno: {allowed_values[:5]}...")
                input_data[col] = 0

    df_input = pd.DataFrame([input_data])
    df_input = df_input[model.feature_names_in_]

    df_scaled = pd.DataFrame(scaler.transform(df_input), columns=model.feature_names_in_)
    return model.predict(df_scaled)[0]

if __name__ == "__main__":
    sample_car = {
        'brand': 'DACIA',
        'year': 2015,
        'mileage': 180000,
        'horsepower': 110,
        'engine_capacity, cc': 1600.0,
        'seats_amount': 5.0,
        'A/C': 'automatic A/C',
        'fuel': 'diesel',
        'car_type': 'hatchback',
        'type_of_drive': 'front',
        'gearbox': 'manual, 6 speeds',
        'doors': '4/5 doors'
    }

    try:
        print("Učitavam model i vršim predikciju...")

        predicted_price = predict_car_price(sample_car)

        print("\n" + "="*30)
        print(f"PROCENA CENE: {predicted_price:,.2f} €")
        print("="*30)

    except KeyError as e:
        print(f"\nGREŠKA: Nedostaje kolona u ulaznim podacima: {e}")
        print("Proveri da li nazivi u 'sample_car' odgovaraju kolonama iz treninga.")
    except Exception as e:
        print(f"\nDošlo je do greške: {e}")
