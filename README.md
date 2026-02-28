# ☀️ Solar Energy Generation Prediction AI

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview
This project focuses on predicting **Solar Power Generation** using machine learning techniques. By analyzing meteorological data (Temperature, Irradiation) and historical power generation logs, the model predicts the **DC Power Output** (kW) of a solar power plant.

This work was developed as part of the **ST5000CEM: Introduction to AI** module at **Softwarica College (In collaboration with Coventry University)**.

### 🎯 Key Objectives
- To solve the problem of solar energy intermittency and grid instability.
- To compare a baseline model (**Linear Regression**) against a non-linear ensemble model (**Random Forest**).
- To optimize the model using **GridSearchCV** and **Cross-Validation**.

---

## 📂 Dataset
The dataset used in this project is sourced from the [Kaggle Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data).

It consists of two files that were merged based on timestamps:
1.  **Plant_1_Generation_Data.csv**: Contains `DC_POWER` (Target) and `AC_POWER`.
2.  **Plant_1_Weather_Sensor_Data.csv**: Contains `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, and `IRRADIATION`.

**Key Features Used:**
- `IRRADIATION`: Solar irradiance (kW/m²).
- `MODULE_TEMPERATURE`: Temperature of the PV module (°C).
- `AMBIENT_TEMPERATURE`: Air temperature (°C).
- `Hour`: Time of day (Derived feature).

---

## 🛠️ Methodology & Algorithms

### 1. Data Preprocessing
- **Merging:** Inner join of Generation and Weather datasets on `DATE_TIME`.
- **Feature Engineering:** Extracted `Hour` from timestamps to capture daily cycles.
- **Cleaning:** Dropped redundant columns (`PLANT_ID`, `SOURCE_KEY`).
- **Scaling:** Applied `StandardScaler` for the Linear Regression baseline.

### 2. Models Implemented
- **Linear Regression (Baseline):** Used to establish a benchmark performance.
- **Random Forest Regressor (Optimized):** An ensemble learning method chosen to handle the non-linear relationship between temperature and power generation efficiency.

### 3. Hyperparameter Tuning
Used **GridSearchCV** with **3-Fold Cross-Validation** to tune:
- `n_estimators`: [50, 100]
- `max_depth`: [10, 20, None]
- `min_samples_split`: [2, 5]

---

## 📊 Results & Evaluation

The models were evaluated using **Root Mean Squared Error (RMSE)** and **R-Squared ($R^2$)**.

| Model | RMSE (kW) | R² Score |
| :--- | :--- | :--- |
| **Linear Regression** | 567.00 | 0.9801 |
| **Random Forest (Optimized)** | **470.54** | **0.9863** |

**Key Findings:**
- The **Random Forest** model outperformed the Linear Regression baseline, reducing the error (RMSE) by nearly **100 kW**.
- **Feature Importance Analysis** revealed that `IRRADIATION` is the most critical predictor, followed by `MODULE_TEMPERATURE`.

---

## 🚀 How to Run the Project

### Prerequisites
Ensure you have Python installed. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Steps
Clone this repository:
```bash
git clone https://github.com/beingbivek/Solar-Prediction-AI.git
```
Navigate to the directory:
```bash
cd Solar-Prediction-AI
```
Run the main script:
```bash
python solarprediction.py
```
📈 Visualizations
(These plots are generated when running the code)

Actual vs. Predicted: Shows how closely the model tracks real power generation.
Feature Importance: Displays which weather factors impact generation the most.
Error Distribution: A histogram showing the spread of prediction errors.

👤 Author

Bivek

📜 License
This project is open-source and available under the MIT License.
