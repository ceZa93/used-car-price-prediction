import joblib
import pandas as pd

def predict_car_price(input_data):
    model = joblib.load('models/best_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    encoders = joblib.load('models/label_encoders.joblib')
    brand_price = joblib.load('models/brand_price_map.joblib')

    # Brend se enkodira po prosečnoj ceni (target encoding iz treninga).
    if 'brand' in input_data:
        brand_key = str(input_data['brand']).upper()
        input_data['brand'] = brand_price['map'].get(brand_key, brand_price['global_mean'])
        if brand_key not in brand_price['map']:
            print(f"Upozorenje: brend '{brand_key}' nije viđen u treningu, koristi se prosečna cena.")

    for col, value in input_data.items():
        if col in encoders:
            # IZBRIŠI .upper() DA BI KORISTIO TAČNO ONO ŠTO UPIŠEŠ
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
        'brand': 'TOYOTA', 
        'mileage': 180000,
        'horsepower': 95,
        'engine_capacity, cc': 1500.0,
        'seats_amount': 5.0,
        'A/C': 'manual A/C',
        'fuel': 'petrol + gas',
        'car_type': 'hatchback',
        'type_of_drive': 'front',
        'gearbox': 'manual, 5 speeds',
        'doors': '4/5 doors',
        'car_age': 9,
        'km_per_year': 20000
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