from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _load_model_and_metadata():
    model = joblib.load('models/best_model.joblib')
    metadata_path = Path('models/training_metadata.joblib')
    metadata = joblib.load(metadata_path) if metadata_path.exists() else {}
    return model, metadata


def _align_input_frame(input_data, expected_columns):
    df_input = pd.DataFrame([dict(input_data)])

    for column in expected_columns:
        if column not in df_input.columns:
            df_input[column] = np.nan

    return df_input.reindex(columns=expected_columns)


def predict_car_price(input_data):
    model, metadata = _load_model_and_metadata()

    expected_columns = metadata.get('selected_features')
    if expected_columns is None:
        expected_columns = list(getattr(model, 'feature_names_in_', []))

    df_input = _align_input_frame(input_data, expected_columns)
    return float(model.predict(df_input)[0])

if __name__ == "__main__":
    sample_car = {
        'car_name': 'Dacia Duster',
        'year': 2015,
        'mileage': 180000,
        'horsepower': 95,
        'engine_capacity, cc': 1500.0,
        'seats_amount': 5.0,
        'A/C': 'manual A/C',
        'emission_class': 'Euro 4',
        'color': 'Black',
        'type_of_drive': 'front',
        'doors': '4/5 doors',
        'fuel': 'petrol',
        'car_type': 'hatchback',
        'gearbox': 'manual, 5 speeds',
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
