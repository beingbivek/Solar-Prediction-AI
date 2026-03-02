import tkinter as tk
from tkinter import ttk, messagebox, Toplevel, Text, Scrollbar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# UI CONFIGURATION
DARK_BG = "#2E2E2E"
LIGHT_TEXT = "#FFFFFF"
ACCENT_COLOR = "#4CAF50"
BUTTON_BG = "#444444"
FONT_MAIN = ("Segoe UI", 12)
FONT_HEADER = ("Segoe UI", 20, "bold")
FONT_MONO = ("Consolas", 10)

class SolarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Solar Energy Prediction System (Random Forest)")
        self.root.geometry("1000x800")
        self.root.configure(bg=DARK_BG)
        
        # Model Variables
        self.model = None
        self.features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'Hour']
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.y_pred = None

        self.setup_styles()
        
        self.container = tk.Frame(self.root, bg=DARK_BG)
        self.container.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.show_home_screen()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background=DARK_BG)
        style.configure("TLabel", background=DARK_BG, foreground=LIGHT_TEXT, font=FONT_MAIN)
        style.configure("Header.TLabel", font=FONT_HEADER, foreground=ACCENT_COLOR)
        style.configure("TButton", font=("Segoe UI", 11, "bold"), background=BUTTON_BG, foreground=LIGHT_TEXT, borderwidth=0, focuscolor=DARK_BG)
        style.map("TButton", background=[('active', ACCENT_COLOR)])
        style.configure("TEntry", fieldbackground="#555555", foreground=LIGHT_TEXT)

    def show_dataframe_popup(self, title, df, note=""):
        """Displays data snapshots."""
        popup = Toplevel(self.root)
        popup.title(title)
        popup.geometry("800x500")
        popup.configure(bg=DARK_BG)
        
        tk.Label(popup, text=title, font=("Segoe UI", 16, "bold"), bg=DARK_BG, fg=ACCENT_COLOR).pack(pady=10)
        if note: tk.Label(popup, text=note, font=("Segoe UI", 10, "italic"), bg=DARK_BG, fg="#AAAAAA").pack(pady=5)
        
        info_str = f"Shape: {df.shape}\n\n--- Head ---\n{df.head().to_string()}"
        
        frame = tk.Frame(popup, bg=DARK_BG)
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        text_area = Text(frame, bg="#1E1E1E", fg="#00FF00", font=FONT_MONO, wrap="none")
        text_area.insert(tk.END, info_str)
        text_area.config(state="disabled")
        text_area.pack(side="left", fill="both", expand=True)
        tk.Button(popup, text="Next Step ➡", command=popup.destroy, bg=ACCENT_COLOR, fg="white", font=("Segoe UI", 12, "bold")).pack(pady=10)
        self.root.wait_window(popup)

    # CORE LOGIC
    def clear_screen(self):
        for widget in self.container.winfo_children(): widget.destroy()

    def create_modern_button(self, parent, text, command):
        return tk.Button(parent, text=text, command=command, font=("Segoe UI", 11), bg=BUTTON_BG, fg=LIGHT_TEXT, activebackground=ACCENT_COLOR, activeforeground=LIGHT_TEXT, bd=0, padx=20, pady=10, width=30, cursor="hand2")

    def show_home_screen(self):
        self.clear_screen()
        tk.Label(self.container, text="☀️ Solar Gen AI", font=("Segoe UI", 32, "bold"), bg=DARK_BG, fg=ACCENT_COLOR).pack(pady=(40, 10))
        tk.Label(self.container, text="Advanced Solar Power Prediction System", font=("Segoe UI", 14), bg=DARK_BG, fg="#AAAAAA").pack(pady=(0, 40))

        btn_frame = tk.Frame(self.container, bg=DARK_BG)
        btn_frame.pack()
        self.create_modern_button(btn_frame, "🚀 Train Model & Generate Report", self.train_model_logic).pack(pady=10, fill='x')
        self.create_modern_button(btn_frame, "📊 Compare K-Fold Graphs", self.compare_k_values).pack(pady=10, fill='x')
        self.create_modern_button(btn_frame, "🔮 Real-time Prediction", self.show_predict_screen).pack(pady=10, fill='x')
        self.create_modern_button(btn_frame, "❌ Exit", self.root.quit).pack(pady=10, fill='x')

    def train_model_logic(self):
        try:
            # 1. Ingestion
            gen_df = pd.read_csv('Plant_1_Generation_Data.csv')
            weather_df = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')
            self.show_dataframe_popup("Step 1: Raw Data", gen_df)

            # 2. Preprocessing
            gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'])
            weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'])
            gen_df.drop(columns=['PLANT_ID', 'SOURCE_KEY', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD'], inplace=True)
            weather_df.drop(columns=['PLANT_ID', 'SOURCE_KEY'], inplace=True)
            
            df = pd.merge(gen_df, weather_df, on='DATE_TIME', how='inner')
            self.show_dataframe_popup("Step 2: Merged Data", df)

            df['Hour'] = df['DATE_TIME'].dt.hour
            self.show_dataframe_popup("Step 3: Cleaned Data (+Hour)", df[self.features + ['DC_POWER']])

            # 3. Splitting
            X = df[self.features]
            y = df['DC_POWER']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # MATH GENERATION (Linear Regression Baseline)
            lr = LinearRegression()
            lr.fit(self.X_train, self.y_train)
            
            beta_0 = lr.intercept_
            betas = lr.coef_
            
            lr_eq = f"y = {beta_0:.2f} "
            for i, feat in enumerate(self.features):
                lr_eq += f"+ ({betas[i]:.2f} * {feat}) "

            # MAIN TRAINING (Random Forest)
            self.model = RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_split=5, random_state=42)
            self.model.fit(self.X_train, self.y_train)
            
            # K-Fold Check (Quick check for K=3)
            cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=3, scoring='r2')
            
            # Evaluate
            self.y_pred = self.model.predict(self.X_test)
            rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
            r2 = r2_score(self.y_test, self.y_pred)

            joblib.dump(self.model, 'solar_model.pkl')
            
            # Construct Technical Report String
            report = f"""
        MODEL MATHEMATICAL ANALYSIS
--------------------------------------------------

1. LINEAR REGRESSION BASELINE (BETA MATRIX)
--------------------------------------------------
Intercept (β0): {beta_0:.4f}

Coefficients (Weights):
 - Ambient Temp (β1): {betas[0]:.4f}
 - Module Temp  (β2): {betas[1]:.4f}
 - Irradiation  (β3): {betas[2]:.4f}
 - Hour         (β4): {betas[3]:.4f}

Full Equation:
{lr_eq}

2. RANDOM FOREST ENSEMBLE (PROPOSED)
--------------------------------------------------
Formula: y = (1/N) * Σ Tree_i(x)

Hyperparameters:
 - n_estimators (Trees): 50
 - max_depth: 20
 - min_samples_split: 5

3. CROSS-VALIDATION RESULTS
--------------------------------------------------
K-Fold Strategy: K=3
Fold Scores: {cv_scores}
Average Accuracy: {cv_scores.mean():.4f}

4. FINAL EVALUATION (TEST SET)
--------------------------------------------------
RMSE: {rmse:.2f} kW
R2 Score: {r2:.4f}
"""
            self.show_training_results_screen(rmse, r2, report)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_training_results_screen(self, rmse, r2, report_text):
        self.clear_screen()
        tk.Label(self.container, text="✅ Model Analysis Complete", font=FONT_HEADER, bg=DARK_BG, fg=ACCENT_COLOR).pack(pady=10)
        
        # Split screen: Left (Metrics/Buttons), Right (Report)
        content_frame = tk.Frame(self.container, bg=DARK_BG)
        content_frame.pack(fill="both", expand=True)
        
        # LEFT SIDE
        left_frame = tk.Frame(content_frame, bg=DARK_BG)
        left_frame.pack(side="left", fill="y", padx=20)
        
        tk.Label(left_frame, text=f"RMSE: {rmse:.2f} kW", font=("Segoe UI", 16, "bold"), bg=DARK_BG, fg="#FF6B6B").pack(anchor="w", pady=10)
        tk.Label(left_frame, text=f"R² Score: {r2:.4f}", font=("Segoe UI", 16, "bold"), bg=DARK_BG, fg="#4CAF50").pack(anchor="w", pady=10)
        
        tk.Label(left_frame, text="Visualizations:", font=("Segoe UI", 12), bg=DARK_BG, fg=LIGHT_TEXT).pack(pady=(20, 5))
        self.create_modern_button(left_frame, "Scatter Plot", lambda: self.plot_graph('scatter')).pack(pady=5)
        self.create_modern_button(left_frame, "Feature Importance", lambda: self.plot_graph('feature')).pack(pady=5)
        self.create_modern_button(left_frame, "Error Histogram", lambda: self.plot_graph('error')).pack(pady=5)
        self.create_modern_button(left_frame, "📉 Linear vs RF (Bias Check)", self.plot_linear_vs_rf).pack(pady=5)
        
        # RIGHT SIDE (Report)
        right_frame = tk.Frame(content_frame, bg="#1E1E1E")
        right_frame.pack(side="right", fill="both", expand=True, padx=20)
        
        tk.Label(right_frame, text="Technical Report:", bg="#1E1E1E", fg=ACCENT_COLOR, font=("Consolas", 10, "bold")).pack(anchor="w")
        
        text_area = Text(right_frame, bg="#1E1E1E", fg="#00FF00", font=("Consolas", 9), wrap="none")
        text_area.insert(tk.END, report_text)
        text_area.pack(side="left", fill="both", expand=True)
        
        scroll = Scrollbar(right_frame, command=text_area.yview)
        scroll.pack(side="right", fill="y")
        text_area.config(yscrollcommand=scroll.set)

        self.create_modern_button(self.container, "⬅ Return Home", self.show_home_screen).pack(pady=20)

    def compare_k_values(self):
        """Generates Figure 4.1"""
        if self.X_train is None:
            messagebox.showwarning("Error", "Train model first.")
            return
            
        k_values = [2, 3, 5, 10]
        results = []
        
        loading = tk.Label(self.container, text="Running Validation...", font=("Segoe UI", 12), bg=DARK_BG, fg="yellow")
        loading.pack()
        self.root.update()

        for k in k_values:
            rf = RandomForestRegressor(n_estimators=20, max_depth=10, random_state=42)
            scores = cross_val_score(rf, self.X_train, self.y_train, cv=k, scoring='r2')
            results.append(scores.mean())
        
        loading.destroy()
        
        plt.figure(figsize=(8, 5))
        sns.lineplot(x=k_values, y=results, marker='o', color='cyan', linewidth=2)
        plt.title("Impact of K-Folds on Accuracy")
        plt.xlabel("K Folds")
        plt.ylabel("R2 Score")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def plot_graph(self, plot_type):
        plt.figure(figsize=(10, 6))
        if plot_type == 'scatter':
            sns.scatterplot(x=self.y_test, y=self.y_pred, alpha=0.3, color='blue')
            plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2, label="Perfect Fit")
            plt.title('Actual vs Predicted')
            plt.xlabel('Actual Power (kW)')
            plt.ylabel('Predicted Power (kW)')
            plt.legend()
        elif plot_type == 'feature':
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            sns.barplot(x=importances[indices], y=[self.features[i] for i in indices], hue=[self.features[i] for i in indices], legend=False, palette='viridis')
            plt.title('Feature Importance')
        elif plot_type == 'error':
            sns.histplot(self.y_test - self.y_pred, kde=True, bins=30, color='purple')
            plt.title('Error Distribution')
        plt.show()

    def plot_linear_vs_rf(self):
        """Generates Side-by-Side Comparison for Bias Analysis"""
        if self.X_train is None: return

        plt.figure(figsize=(14, 6))

        # Subplot 1: Linear Regression (High Bias)
        plt.subplot(1, 2, 1)
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)
        lr_pred = lr.predict(self.X_test)
        
        sns.scatterplot(x=self.y_test, y=lr_pred, alpha=0.3, color='red')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=2)
        plt.title('Linear Regression (High Bias)')
        plt.xlabel('Actual Power')
        plt.ylabel('Predicted Power')

        # Subplot 2: Random Forest (Low Bias)
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=self.y_test, y=self.y_pred, alpha=0.3, color='blue')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'k--', lw=2)
        plt.title('Random Forest (Low Bias)')
        plt.xlabel('Actual Power')
        plt.ylabel('Predicted Power')

        plt.tight_layout()
        plt.show()

    def show_predict_screen(self):
        if not os.path.exists('solar_model.pkl'): 
            messagebox.showwarning("Error", "Train model first.")
            return
        self.model = joblib.load('solar_model.pkl')
        self.clear_screen()
        tk.Label(self.container, text="🔮 Predict", font=FONT_HEADER, bg=DARK_BG, fg=ACCENT_COLOR).pack(pady=20)
        
        form_frame = tk.Frame(self.container, bg=DARK_BG)
        form_frame.pack()
        self.entries = {}
        for i, feat in enumerate(self.features):
            tk.Label(form_frame, text=feat, bg=DARK_BG, fg=LIGHT_TEXT).grid(row=i, column=0, sticky="w", pady=5)
            entry = ttk.Entry(form_frame)
            entry.grid(row=i, column=1, pady=5)
            self.entries[feat] = entry
            
        self.result_label = tk.Label(self.container, text="", font=("Segoe UI", 16), bg=DARK_BG, fg="#FFD700")
        self.result_label.pack(pady=10)
        self.create_modern_button(self.container, "Predict", self.perform_prediction).pack()
        self.create_modern_button(self.container, "Back", self.show_home_screen).pack(pady=10)

    def perform_prediction(self):
        try:
            inputs = [float(self.entries[f].get()) for f in self.features]
            input_df = pd.DataFrame([inputs], columns=self.features)
            pred = self.model.predict(input_df)[0]
            self.result_label.config(text=f"Result: {pred:.2f} kW")
        except: messagebox.showerror("Error", "Invalid Input")

if __name__ == "__main__":
    root = tk.Tk()
    app = SolarApp(root)
    root.mainloop()