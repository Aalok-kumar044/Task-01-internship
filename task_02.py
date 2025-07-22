import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd

# === Realistic dummy dataset ===
data = pd.DataFrame({
    'Mileage': [15, 18, 20, 12, 14, 22, 16, 13, 19, 21],
    'Condition': [4, 5, 3, 2, 4, 5, 3, 2, 4, 5],
    'Grade': [4, 5, 3, 2, 4, 5, 3, 2, 4, 5],
    'RunningKM': [50000, 30000, 70000, 90000, 45000, 25000, 80000, 100000, 60000, 20000],
    'Price': [500000, 700000, 450000, 300000, 520000, 750000, 400000, 280000, 600000, 800000]
})

# === Model training ===
X = data[['Mileage', 'Condition', 'Grade', 'RunningKM']]
y = data['Price']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# === Prediction and chart drawing ===
def predict_price():
    try:
        mileage = float(mileage_entry.get())
        condition = float(condition_cb.get())
        grade = float(grade_cb.get())
        running_km = float(running_km_entry.get())

        features = np.array([[mileage, condition, grade, running_km]])
        predicted_price = model.predict(features)[0]

        result_label.config(text=f"Estimated Price: ₹{int(predicted_price):,}")

        draw_chart(predicted_price)

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers.")

# === Drawing chart ===
def draw_chart(predicted_price):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(data['RunningKM'], data['Price'], color='blue', label='Actual Cars')
    ax.scatter(float(running_km_entry.get()), predicted_price, color='red', label='Predicted Car', s=100)
    ax.set_title("Car Price Comparison")
    ax.set_xlabel("Running KM")
    ax.set_ylabel("Price")
    ax.legend()

    for widget in chart_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# === GUI Setup ===
root = tk.Tk()
root.title("AI Car Price Predictor")
root.geometry("600x600")
style = ttk.Style(root)
style.theme_use("clam")

main_frame = ttk.Frame(root, padding="10")
main_frame.pack(padx=10, pady=10, fill='both', expand=True)

# Inputs
ttk.Label(main_frame, text="Mileage (km/l):").grid(row=0, column=0, sticky='e', padx=5, pady=5)
mileage_entry = ttk.Entry(main_frame)
mileage_entry.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(main_frame, text="Condition (1-5):").grid(row=1, column=0, sticky='e', padx=5, pady=5)
condition_cb = ttk.Combobox(main_frame, values=[1, 2, 3, 4, 5], state='readonly')
condition_cb.grid(row=1, column=1, padx=5, pady=5)
condition_cb.current(3)

ttk.Label(main_frame, text="Grade (1-5):").grid(row=2, column=0, sticky='e', padx=5, pady=5)
grade_cb = ttk.Combobox(main_frame, values=[1, 2, 3, 4, 5], state='readonly')
grade_cb.grid(row=2, column=1, padx=5, pady=5)
grade_cb.current(3)

ttk.Label(main_frame, text="Running KM:").grid(row=3, column=0, sticky='e', padx=5, pady=5)
running_km_entry = ttk.Entry(main_frame)
running_km_entry.grid(row=3, column=1, padx=5, pady=5)

predict_button = ttk.Button(main_frame, text="Predict Price", command=predict_price)
predict_button.grid(row=4, column=0, columnspan=2, pady=10)

result_label = ttk.Label(main_frame, text="Estimated Price: ₹--", font=("Arial", 14))
result_label.grid(row=5, column=0, columnspan=2, pady=10)

# Chart Frame
chart_frame = ttk.Frame(root)
chart_frame.pack(padx=10, pady=10, fill='both', expand=True)

root.mainloop()