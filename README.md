# ☀️ Solar Energy Generation Prediction System (AI + GUI)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Tkinter-green)
![Library](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Project Overview
This project is an advanced AI-powered system designed to predict **Solar Power Generation** based on meteorological data. It features a **Modern Dark-Themed GUI** built with Python's Tkinter, allowing users to train the model, visualize results, and make real-time predictions without writing code.

This work was developed as part of the **ST5000CEM: Introduction to AI** module at **Softwarica College (Coventry University)**.

### 🎯 Key Features
- **Modern GUI:** A clean, dark-themed user interface for easy interaction.
- **One-Click Training:** Automatically merges datasets, preprocesses data, and trains a **Random Forest Regressor**.
- **Model Persistence:** Saves the trained model (`solar_model.pkl`) for future use.
- **Interactive Visualizations:** View Actual vs. Predicted plots, Feature Importance, and Error Distribution histograms directly from the app.
- **Real-Time Prediction:** Input custom weather parameters (Temperature, Irradiation, Hour) to get an instant power generation forecast (kW).

---

## 📂 Dataset
The system requires two CSV files from the [Kaggle Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data):
1.  **Plant_1_Generation_Data.csv**: Contains `DC_POWER` (Target) and timestamps.
2.  **Plant_1_Weather_Sensor_Data.csv**: Contains `AMBIENT_TEMPERATURE`, `MODULE_TEMPERATURE`, and `IRRADIATION`.

**Preprocessing:**
- The app automatically merges these files based on the `DATE_TIME` column.
- It extracts the `Hour` feature to capture daily solar cycles.

---

## 🛠️ Methodology & Algorithms

### 1. The Algorithm
- **Random Forest Regressor:** Selected for its ability to handle non-linear relationships (e.g., efficiency drop at high temperatures).
- **Comparison:** Outperformed Linear Regression (Baseline) in our tests.

### 2. Hyperparameter Tuning
Optimized via **GridSearchCV** with **3-Fold Cross-Validation**:
- `n_estimators`: 50
- `max_depth`: 20
- `min_samples_split`: 5

---

## 🚀 Installation & Usage

### Prerequisites
Ensure you have Python installed. Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
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
python solar-prediction-GUI.py
```
## 📊 Performance Results

The model was evaluated using **Root Mean Squared Error (RMSE)** and **R-Squared ($R^2$)**.

| Metric | Value |
| :--- | :--- |
| **Root Mean Squared Error (RMSE)** | ~470.54 kW |
| **R-Squared ($R^2$) Score** | 0.9863 |

> **Note:** Results may vary slightly depending on the random seed during training.

👤 Author - Bivek Thapa
🪪 CU ID - 15938381

📜 License
This project is open-source and available under the MIT License.