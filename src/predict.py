import logging

import joblib
import pandas as pd

REFERENCE_YEAR = 2024

logger = logging.getLogger(__name__)

_REQUIRED_ARTIFACTS = [
    'models/best_model.joblib',
    'models/scaler.joblib',
    'models/label_encoders.joblib',
    'models/brand_price_map.joblib',
]


def _load_artifact(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Required model artifact not found: '{path}'. "
            "Run the training pipeline (main.py) first."
        )


def predict_car_price(input_data):
    model = _load_artifact('models/best_model.joblib')
    scaler = _load_artifact('models/scaler.joblib')
    encoders = _load_artifact('models/label_encoders.joblib')
    brand_price = _load_artifact('models/brand_price_map.joblib')

    input_data = dict(input_data)

    if 'year' in input_data and 'car_age' not in input_data:
        input_data['car_age'] = REFERENCE_YEAR - int(input_data['year'])
    input_data.pop('year', None)

    if 'km_per_year' not in input_data and 'mileage' in input_data and 'car_age' in input_data:
        age = input_data['car_age'] if input_data['car_age'] != 0 else 1
        input_data['km_per_year'] = input_data['mileage'] / age

    if 'brand' in input_data:
        brand_key = str(input_data['brand']).upper()
        if brand_key not in brand_price['map']:
            logger.warning(
                "Brand '%s' was not seen during training — using global mean price.",
                brand_key,
            )
        input_data['brand'] = brand_price['map'].get(brand_key, brand_price['global_mean'])

    for col, value in input_data.items():
        if col in encoders:
            val_str = str(value)
            allowed_values = encoders[col].classes_

            if val_str in allowed_values:
                input_data[col] = encoders[col].transform([val_str])[0]
            else:
                raise ValueError(
                    f"Unknown value '{val_str}' for column '{col}'. "
                    f"Allowed values (first 10): {list(allowed_values[:10])}"
                )

    df_input = pd.DataFrame([input_data])

    if not hasattr(model, 'feature_names_in_'):
        raise AttributeError(
            "Loaded model does not expose 'feature_names_in_'. "
            "Re-train using a scikit-learn version that supports this attribute."
        )

    missing_features = set(model.feature_names_in_) - set(df_input.columns)
    if missing_features:
        raise ValueError(
            f"Input is missing required features: {sorted(missing_features)}. "
            f"Expected features: {list(model.feature_names_in_)}"
        )

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
        'doors': '4/5 doors',
    }

    try:
        print("Ucitavam model i vrsim predikciju...")

        predicted_price = predict_car_price(sample_car)

        print("\n" + "=" * 30)
        print(f"PROCENA CENE: {predicted_price:,.2f} EUR")
        print("=" * 30)

    except FileNotFoundError as e:
        print(f"\nGRESKA: {e}")
    except ValueError as e:
        print(f"\nGRESKA u ulaznim podacima: {e}")
    except Exception as e:
        print(f"\nNeocekivana greska: {e}")
