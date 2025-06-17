# Inventory Forecasting with LSTM

A production‑ready, end‑to‑end machine‑learning pipeline that predicts **next‑day inventory ending quantity** from historical consumption data. The project couples an LSTM time‑series model with a Flask micro‑service so you can serve predictions through a lightweight web API or browser‑based form.

---

## :clipboard: Table of contents
1. [Project overview](#project-overview)
2. [Repository structure](#repository-structure)
3. [Quick start](#quick-start)
4. [Usage](#usage)
   * [Web UI](#web-ui)
   * [REST API](#rest-api)
5. [Data & preprocessing](#data--preprocessing)
6. [Model development](#model-development)
7. [Retraining the model](#retraining-the-model)
8. [Tests](#tests)
9. [Contributing](#contributing)
10. [License](#license)

---

## Project overview

**Problem**  
Restaurant managers need to know tomorrow’s *Ending Quantity* of each ingredient so they can place the right purchase order—minimising wastage while avoiding stock‑outs.

**Solution**  
We train a univariate LSTM that consumes the last **30 days of consumption** per ingredient and outputs the following day’s ending quantity. The final pipeline consists of:

| Component | Purpose |
|-----------|---------|
| `inventory_data_large.csv` | Four‑year transactional dataset (508k rows) spanning 12 locations × 29 ingredients, 2018‑01‑01 → 2021‑12‑31. |
| `Model Development.ipynb` | Research notebook: EDA, feature engineering, hyper‑parameter tuning, walk‑forward validation. |
| `inventory_model.keras` | SavedModel frozen after best validation RMSE (see notebook for metrics). |
| `scaler.pkl` | `sklearn.preprocessing.StandardScaler` fitted on train set; used to scale both inputs & predictions. |
| `app.py` | Flask service exposing an HTML form (`/`) and a JSON endpoint (`/predict`). |

---

## Repository structure

```
.
├── app.py
├── inventory_model.keras/
├── scaler.pkl
├── inventory_data_large.csv
├── Model Development.ipynb
├── requirements.txt
└── templates/
```

> **Heads‑up:** The `templates/` folder is required by Flask. If you’re only using the REST API, you can ignore it.

---

## Quick start

### 1 · Clone & set up the environment

```bash
git clone https://github.com/your-org/inventory-forecast-lstm.git
cd inventory-forecast-lstm

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2 · Run the web service

```bash
python app.py
# → Listening on http://127.0.0.1:5000
```

To run in production:

```bash
gunicorn -b 0.0.0.0:5000 app:app
```

---

## Usage

### Web UI

Visit: `http://localhost:5000`  
Enter *today’s consumption*. The app returns the *predicted ending quantity*.

### REST API

```bash
curl -X POST -F "consumption=135" http://localhost:5000/predict
```

Returns:

```json
{ "prediction": 487.32 }
```

---

## Data & preprocessing

| Column | Description |
|--------|-------------|
| `Date` | Daily calendar date |
| `Day` | Day‑of‑week label |
| `Month` | Month label (e.g. Jan‑18) |
| `Location` | Restaurant site |
| `Ingredient` | Raw material |
| `Starting_Quantity` | Units in stock at start of day |
| `Purchased` | Units bought |
| `Consumption` | Units used (target) |
| `Ending_Quantity` | Next-day starting stock |
| `Status` | Operational status |

### Commentary

- Data is aggregated per ingredient.
- Nulls and outliers are handled using filters and sanity checks.
- Scaled with `StandardScaler` to improve convergence.
- Inputs: 30 days of `Consumption`; Output: next-day `Ending_Quantity`.

### Source

This dataset is synthetic and was created to simulate real inventory patterns.  
For real-world experimentation, refer to:

**FreshRetailNet-50K**:  
https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K

---

## Model development

- **LSTM** with 64 units → Dense(1)
- **Input window**: 30 days
- **Loss**: MSE
- **Optimizer**: Adam (lr = 0.001)
- **Validation**: walk-forward strategy

See `Model Development.ipynb` for full tuning and evaluation metrics.

---

## Retraining the model

1. Run all cells in `Model Development.ipynb`
2. Save model:

```python
model.save("inventory_model.keras")
```

3. Refit `StandardScaler` on full data and export to `scaler.pkl`

---

## Tests

```bash
pytest -q
```

Tests include:
- Input validation
- Shape/scale matching
- Endpoint health

---

## Contributing

- Fork the repo and create a feature branch.
- Push atomic commits with clear messages.
- Open a pull request against `main`.

---

## License

MIT License. See `LICENSE` file.
