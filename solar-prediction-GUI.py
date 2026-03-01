import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# CONFIGURATION & STYLE
# ==========================================
DARK_BG = "#2E2E2E"      # Dark Grey Background
LIGHT_TEXT = "#FFFFFF"   # White Text
ACCENT_COLOR = "#4CAF50" # Green Accent
BUTTON_BG = "#444444"    # Button Background
FONT_MAIN = ("Segoe UI", 12)
FONT_HEADER = ("Segoe UI", 20, "bold")

class SolarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Solar Energy Prediction AI")
        self.root.geometry("800x600")
        self.root.configure(bg=DARK_BG)
        
        # Variables to store model data
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'Hour']

        # Setup Styles
        self.setup_styles()
        
        # Container for changing screens
        self.container = tk.Frame(self.root, bg=DARK_BG)
        self.container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Start with Home Screen
        self.show_home_screen()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure Frame, Label, Button styles
        style.configure("TFrame", background=DARK_BG)
        style.configure("TLabel", background=DARK_BG, foreground=LIGHT_TEXT, font=FONT_MAIN)
        style.configure("Header.TLabel", font=FONT_HEADER, foreground=ACCENT_COLOR)
        
        style.configure("TButton", 
                        font=("Segoe UI", 11, "bold"), 
                        background=BUTTON_BG, 
                        foreground=LIGHT_TEXT, 
                        borderwidth=0, 
                        focuscolor=DARK_BG)
        style.map("TButton", background=[('active', ACCENT_COLOR)]) # Hover effect

        style.configure("TEntry", fieldbackground="#555555", foreground=LIGHT_TEXT)

    # ==========================================
    # SCREENS
    # ==========================================

    def clear_screen(self):
        for widget in self.container.winfo_children():
            widget.destroy()

    def show_home_screen(self):
        self.clear_screen()
        
        # Title
        tk.Label(self.container, text="☀️ Solar Gen AI", font=("Segoe UI", 32, "bold"), bg=DARK_BG, fg=ACCENT_COLOR).pack(pady=(40, 10))
        tk.Label(self.container, text="Advanced Solar Power Prediction System", font=("Segoe UI", 14), bg=DARK_BG, fg="#AAAAAA").pack(pady=(0, 40))

        # Buttons
        btn_frame = tk.Frame(self.container, bg=DARK_BG)
        btn_frame.pack()

        self.create_modern_button(btn_frame, "🚀 Train Model", self.train_model_logic).pack(pady=10, fill='x')
        self.create_modern_button(btn_frame, "🔮 Predict User Input", self.show_predict_screen).pack(pady=10, fill='x')
        self.create_modern_button(btn_frame, "❌ Exit", self.root.quit).pack(pady=10, fill='x')

    def show_training_results_screen(self, rmse, r2):
        self.clear_screen()
        
        tk.Label(self.container, text="✅ Model Trained Successfully!", font=FONT_HEADER, bg=DARK_BG, fg=ACCENT_COLOR).pack(pady=20)
        
        # Results metrics
        result_frame = tk.Frame(self.container, bg="#333333", padx=20, pady=20)
        result_frame.pack(pady=20)
        
        tk.Label(result_frame, text=f"RMSE Error: {rmse:.2f} kW", font=("Segoe UI", 14, "bold"), bg="#333333", fg="#FF6B6B").pack(anchor="w")
        tk.Label(result_frame, text=f"R² Accuracy: {r2:.4f}", font=("Segoe UI", 14, "bold"), bg="#333333", fg="#4CAF50").pack(anchor="w")
        
        tk.Label(self.container, text="Visualizations:", font=("Segoe UI", 12, "bold"), bg=DARK_BG, fg=LIGHT_TEXT).pack(pady=(20, 10))

        # Visualization Buttons
        btn_frame = tk.Frame(self.container, bg=DARK_BG)
        btn_frame.pack()
        
        col1 = tk.Frame(btn_frame, bg=DARK_BG)
        col1.pack(side="left", padx=10)
        col2 = tk.Frame(btn_frame, bg=DARK_BG)
        col2.pack(side="left", padx=10)

        self.create_modern_button(col1, "Actual vs Predicted", lambda: self.plot_graph('scatter')).pack(pady=5, fill='x')
        self.create_modern_button(col1, "Feature Importance", lambda: self.plot_graph('feature')).pack(pady=5, fill='x')
        self.create_modern_button(col2, "Error Distribution", lambda: self.plot_graph('error')).pack(pady=5, fill='x')

        # Back Button
        self.create_modern_button(self.container, "⬅ Back to Home", self.show_home_screen).pack(pady=30)

    def show_predict_screen(self):
        # Check if model exists
        if not os.path.exists('solar_model.pkl'):
            messagebox.showwarning("Model Not Found", "Please 'Train Model' first!")
            return
        
        # Load model if not in memory
        if self.model is None:
            self.model = joblib.load('solar_model.pkl')

        self.clear_screen()
        tk.Label(self.container, text="🔮 Predict Solar Output", font=FONT_HEADER, bg=DARK_BG, fg=ACCENT_COLOR).pack(pady=20)

        # Form Frame
        form_frame = tk.Frame(self.container, bg=DARK_BG)
        form_frame.pack(pady=10)

        # Entry Fields
        self.entries = {}
        labels = {
            'AMBIENT_TEMPERATURE': "Ambient Temp (°C)", 
            'MODULE_TEMPERATURE': "Module Temp (°C)", 
            'IRRADIATION': "Irradiation (kW/m²)", 
            'Hour': "Hour of Day (0-23)"
        }

        for i, (key, text) in enumerate(labels.items()):
            tk.Label(form_frame, text=text, bg=DARK_BG, fg=LIGHT_TEXT).grid(row=i, column=0, sticky="w", pady=5, padx=10)
            entry = ttk.Entry(form_frame, width=25)
            entry.grid(row=i, column=1, pady=5, padx=10)
            self.entries[key] = entry

        # Predict Button
        self.create_modern_button(self.container, "⚡ Calculate Power", self.perform_prediction).pack(pady=20)

        # Result Label
        self.result_label = tk.Label(self.container, text="", font=("Segoe UI", 16, "bold"), bg=DARK_BG, fg="#FFD700")
        self.result_label.pack(pady=10)

        # Back Button
        self.create_modern_button(self.container, "⬅ Back to Home", self.show_home_screen).pack(pady=10)

    # ==========================================
    # LOGIC
    # ==========================================

    def create_modern_button(self, parent, text, command):
        return tk.Button(parent, text=text, command=command, 
                         font=("Segoe UI", 11), bg=BUTTON_BG, fg=LIGHT_TEXT, 
                         activebackground=ACCENT_COLOR, activeforeground=LIGHT_TEXT,
                         bd=0, padx=20, pady=10, width=20, cursor="hand2")

    def train_model_logic(self):
        try:
            # Show loading status (simple)
            loading_lbl = tk.Label(self.container, text="Training Model... Please Wait...", font=("Segoe UI", 14), bg=DARK_BG, fg="#F1C40F")
            loading_lbl.pack()
            self.root.update() # Force UI update

            # 1. Load Data
            gen_df = pd.read_csv('Plant_1_Generation_Data.csv')
            weather_df = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')

            # 2. Preprocessing
            gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'], dayfirst=True)
            weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'], dayfirst=True)
            
            gen_df.drop(columns=['PLANT_ID', 'SOURCE_KEY', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD'], inplace=True)
            weather_df.drop(columns=['PLANT_ID', 'SOURCE_KEY'], inplace=True)
            
            df = pd.merge(gen_df, weather_df, on='DATE_TIME', how='inner')
            df['Hour'] = df['DATE_TIME'].dt.hour
            
            X = df[self.features]
            y = df['DC_POWER']

            # 3. Split
            X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # 4. Train (Using Best Params found earlier to be fast)
            # Best Params: {'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 50}
            self.model = RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_split=5, random_state=42)
            self.model.fit(X_train, y_train)

            # 5. Evaluate
            self.y_pred = self.model.predict(self.X_test)
            rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
            r2 = r2_score(self.y_test, self.y_pred)

            # 6. Save Model
            joblib.dump(self.model, 'solar_model.pkl')

            # Go to Results Screen
            self.show_training_results_screen(rmse, r2)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            self.show_home_screen()

    def plot_graph(self, plot_type):
        if self.model is None: return

        plt.figure(figsize=(10, 6))
        
        if plot_type == 'scatter':
            sns.scatterplot(x=self.y_test, y=self.y_pred, alpha=0.3, color='blue')
            plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            plt.title('Actual vs Predicted Power')
            plt.xlabel('Actual Power (kW)')
            plt.ylabel('Predicted Power (kW)')
            
        elif plot_type == 'feature':
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            sns.barplot(x=importances[indices], y=[self.features[i] for i in indices], palette='viridis')
            plt.title('Feature Importance')
            plt.xlabel('Importance Score')
            
        elif plot_type == 'error':
            residuals = self.y_test - self.y_pred
            sns.histplot(residuals, kde=True, bins=30, color='purple')
            plt.axvline(x=0, color='red', linestyle='--')
            plt.title('Error Distribution (Residuals)')
            plt.xlabel('Error (kW)')

        plt.tight_layout()
        plt.show()

    def perform_prediction(self):
        try:
            # Get values from entries
            inputs = []
            for feat in self.features:
                val = self.entries[feat].get()
                if not val:
                    messagebox.showwarning("Missing Input", f"Please enter {feat}")
                    return
                inputs.append(float(val))
            
            # Create DataFrame with Correct Column Names (Fixes Warning)
            input_df = pd.DataFrame([inputs], columns=self.features)
            
            # Predict
            prediction = self.model.predict(input_df)[0]
            
            # Update Label
            self.result_label.config(text=f"Predicted Power: {prediction:.2f} kW")
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers only.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    app = SolarApp(root)
    root.mainloop()