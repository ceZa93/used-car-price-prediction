# Predikcija cena polovnih automobila — Srpski dataset 2024

Regresioni ML model za predikciju cena polovnih automobila na osnovu karakteristika vozila. Dataset: Srpski automobili (2024) — ~8400 vozila, 14+ atributa.

**Cilj:** Predvidjeti kontinualnu vrednost (`price`) na osnovu ulaznih atributa. Target u modelu: `log1p(price)` — log transformacija stabilizuje jako asimetričnu raspodelu cena; predikcije se vraćaju u eure pomoću `expm1`.

**Finalni rezultat:** MAE = **932€** (greška od ±932€ u proseku)

## Karakteristike dataset-a

**Raw dataset:** 8,413 vozila

| Atribut | Raspon |
|---------|--------|
| **Cijena** | 100-82,000€ |
| **Godina** | 1960-2024 |
| **Kilometraža** | 1-4.3B km |
| **Konjske snage** | 50-485 KS |
| **Zapremina motora** | 100-10,000 cc |
| **Model automobila** | 100+ modela |

## Struktura projekta

```
used-car-price-prediction/
├── data/
│   ├── raw/                      ← originalni sirovi dataset (serbia_car_sales_price_2024_v2.csv)
│   └── processed/                ← pripremljeni dataset nakon čišćenja
├── src/
│   ├── data_preparation.py       ← čišćenje, clipiranje outliers, popunjavanje NaN
│   ├── feature_engineering.py    ← skaliranje year-a, ekstraktovanje brand iz car_name
│   ├── train.py                  ← treniranje modela, GridSearchCV tuning, feature selection
│   ├── evaluate.py               ← EDA grafici, model evaluacija, metrieke
│   └── predict.py                ← predikcija za nove primere
├── models/
│   ├── best_model.joblib         ← finalni Random Forest/Gradient Boosting model
│   ├── training_metadata.joblib  ← selected features i hyperparameteri
│   ├── feature_importance.csv    ← permutation importance (%)
│   ├── model_results.csv         ← poređenje svih modela
│   └── test_metrics.json         ← finalne metrike na test skupu
├── results/
│   ├── figures/                  ← grafici (.png): korelacije, distribucije, važnost features
│   └── metrics/                  ← CSV statistike
├── app/
│   └── ui.py                     ← Streamlit web aplikacija za interaktivne predikcije
├── main.py                       ← glavna skripta (pokreće sve korake)
└── README.md
```

## Feature Importance (Finalni rezultati)

Permutation importance na test skupu:

| Feature | Važnost (%) | Napomena |
|---------|-------------|----------|
| **year** | 52.1% | Najvažniji - starost auta je ključan faktor |
| **car_name** | 24.9% | Model automobila (Dacia, Alfa, Toyota, itd.) |
| **gearbox** | 6.4% | Ručni vs automatski menjač |
| **horsepower** | 4.3% | Snaga motora u KS |
| **engine_capacity, cc** | 2.5% | Zapremina motora |
| Ostali | ~10% | color, fuel, car_type, doors, itd. |

**Objašnjenje:**
- `year` je dominantan jer je starost automobila **jaki linearni prediktor** cene (koreliacija 0.67)
- `car_name` je važan jer različiti modeli imaju različite cijene čak i iste godine (Dacia vs Mercedes)
- `horsepower` i `engine_capacity` su nižerangirani jer su parcijalno korelirani sa `year`

## Redosled operacija (bez curenja podataka)

### 1. **Data Preparation** (`data_preparation.py`)

Operacije koje se dešavaju **pre split-a**:
- Normalizacija tekstualnih kolona (trim, zamena praznih sa NaN)
- Popunjavanje nedostajućih vrednosti:
  - Numeričke: **median**
  - Tekstualne: **'Unknown'**
- **Clipiranje outliers-a:**
  - Cijena: 100-25,000€ (izbacuje ~413 skupih vozila)
  - Kilometraža: 0-550,000 km (koriguje overflow vrijednosti)
  - Horsepower: 50-280 KS (99th percentile)
  - Engine capacity: 500-3,000 cc (99th percentile)
  - Godina: 1980-2024
- Uklanjanje nepotrebnih kolona: `post_info`, `views`, `favorite`, `emission_class`

### 2. **Split 70/15/15** (sa `random_state=42`)

```
Ukupno: ~7,900 vozila
├── Train: ~5,530 (70%) ← GridSearchCV hyperparameter tuning
├── Validation: ~1,185 (15%) ← odabir best modela
└── Test: ~1,185 (15%) ← KORISTI SE SAMO JEDNOM na kraju
```

### 3. **Feature Engineering** (`feature_engineering.py`)

- **Year skaliranje:** `(year - 1980) / (2024 - 1980)` → raspon 0-1
  - 1980 = 0.0 (najstariji), 2024 = 1.0 (najnoviji)
- **Numeričke kolone:** Očuvan originalni format (MinMaxScaler se primenjuje kasnije)
- **Car_name:** Zadržava se za target encoding

### 4. **Preprocessing Pipeline**

```python
Pipeline(
    ('feature_engineering', FeatureEngineer()),
    ('preprocessor', ColumnTransformer(
        ('num', MinMaxScaler(), numeričke_kolone),
        ('cat', Target-Encoder → MinMaxScaler(), kategoričke_kolone)
    )),
    ('model', Random Forest / Gradient Boosting)
)
```

- **Numeričke:** MinMaxScaler (0-1 raspon)
- **Kategoričke (car_name, gearbox, itd.):** 
  - CategoryTargetEncoder (prosječna cijena po vrijednosti)
  - Zatim MinMaxScaler (0-1 raspon)

### 5. **Hyperparameter Tuning** (GridSearchCV, CV=3)

Modeli i njihovi parametri:

| Model | Hiperparametri |
|-------|----------------|
| **Gradient Boosting** | n_estimators: [200, 400], learning_rate: [0.03, 0.05] |
| **Random Forest** | n_estimators: [250, 400], max_depth: [None, 18] |
| **Ridge** | alpha: [0.3, 1.0, 3.0, 10.0] |
| Extra Trees | n_estimators: [300], max_depth: [18] |
| Hist Gradient Boosting | learning_rate: [0.05, 0.1] |

### 6. **Feature Selection**

Permutation importance na validacionom skupu → **odabir top 12 features**. Poređenje:
- `all_features` (13 features)
- `top_12_features` (odabrane)

Odabira se onaj sa **boljim MAE**.

### 7. **Finalni test** na TEST skupu (samo jednom!)

## Rezultati evaluacije

### Poređenje modela (validation skup)

| Model | MAE | RMSE | R² | Median AE |
|-------|-----|------|-----|-----------|
| Ridge | ~1,200€ | ~2,100€ | 0.72 | - |
| **Random Forest** | **~950€** | **~1,700€** | **0.83** | ~566€ |
| Gradient Boosting | ~1,100€ | ~1,900€ | 0.76 | - |
| Extra Trees | ~980€ | ~1,750€ | 0.79 | - |
| Hist Gradient Boosting | ~1,050€ | ~1,850€ | 0.77 | - |

### Finalni rezultat na TEST skupu

**Odabrani model:** Random Forest

| Metrika | Vrijednost |
|---------|-----------|
| MAE (Mean Absolute Error) | **932€** |
| RMSE (Root Mean Squared Error) | ~1,598€ |
| R² (coefficient of determination) | 0.838 |
| Median AE | **566€** |
| Mean Error % | 37.87% |

**Interpretacija:**
- Model predviđa cijenu sa greškom od **±932€** u proseku
- Polovina predikcija ima grešku < **566€**
- 83% varijabilnosti cijena je objašnjeno modelom

## Instalacija i pokretanje

### 1. Setup

```bash
git clone <https://github.com/ceZa93/used-car-price-prediction>
cd used-car-price-prediction
pip install .
```

### 2. Kompletan pipeline (sve odjednom)

```bash
python main.py
```

Generiše:
- Čišćene podatke u `data/processed/`
- Grafike u `results/figures/`
- Model u `models/best_model.joblib`
- Metrike u `models/` i `results/metrics/`

### 3. Streamlit Web UI (interaktivna predikcija)

```bash
streamlit run app/ui.py
```

**Korisnik bira:**
- **Model automobila** (npr. "Dacia Duster", "Alfa Romeo 156")
- Godina proizvodnje (1950-2024)
- Kilometraža, konjske snage, zapremina motora
- Boja, gorivo, menjač, klima, itd.

**Model vraca:** Procenjenu cijenu u eurima

## Tehnologije

- **Python 3.10+**
- **scikit-learn:** Modeliranje, preprocessing, feature selection
- **pandas, numpy:** Manipulacija podacima
- **joblib:** Serijalizacija modela
- **Streamlit:** Web interfejs za predikcije
- **matplotlib, seaborn:** Vizuelizacija