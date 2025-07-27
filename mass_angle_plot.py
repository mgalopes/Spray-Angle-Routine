import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === CONFIG ===
CSV_FOLDER = '.'  # current folder
USE_STD = True    # True = standard deviation; False = mean absolute deviation
# =====================

def extract_temperature(csv_name):
    """Extract temperature like '25C' or '40C' from filename."""
    match = re.search(r'_(\d{2}C)_\d+bar', csv_name)
    return match.group(1) if match else 'Unknown'

def extract_pressure(csv_name):
    """Extract pressure like '50bar', '60bar', '70bar' from filename."""
    match = re.search(r'_(\d+)bar', csv_name)
    return match.group(1) + 'bar' if match else 'Unknown'

# Find CSV files
csv_files = [f for f in os.listdir(CSV_FOLDER) if f.lower().endswith('.csv')]

# We'll collect data as: data[temp][pressure] = list of angles
data = {}

for file in sorted(csv_files):
    path = os.path.join(CSV_FOLDER, file)
    try:
        df = pd.read_csv(path)
        df = df[pd.to_numeric(df['Angle (degrees)'], errors='coerce').notnull()]
        angles = df['Angle (degrees)'].astype(float)

        mean_angle = angles.mean()
        deviation = angles.std() if USE_STD else (angles - mean_angle).abs().mean()

        temp = extract_temperature(file)
        pressure = extract_pressure(file)

        if temp not in data:
            data[temp] = {}
        data[temp][pressure] = (mean_angle, deviation)

    except Exception as e:
        print(f"[!] Failed to process {file}: {e}")

# Define the order of pressures on x-axis
pressure_order = ['50bar', '60bar', '70bar']

# Prepare plot
plt.figure(figsize=(10, 6))
colors = plt.get_cmap('tab10')
temps = sorted(data.keys())

for i, temp in enumerate(temps):
    means = []
    errors = []
    xs = []

    for j, pressure in enumerate(pressure_order):
        if pressure in data[temp]:
            mean_val, err_val = data[temp][pressure]
            means.append(mean_val)
            errors.append(err_val)
            xs.append(j)
        else:
            # If no data for this pressure, skip or use NaN
            means.append(np.nan)
            errors.append(0)
            xs.append(j)

    plt.errorbar(xs, means, yerr=errors, fmt='o--', label=temp, color=colors(i), capsize=5)

plt.xticks(range(len(pressure_order)), pressure_order)
plt.xlabel('Pressure')
plt.ylabel('Spray Angle (degrees)')
plt.title('Spray Angle vs Pressure by Temperature')
plt.grid(True, axis='y', linestyle='--')
plt.legend(title='Temperature')
plt.tight_layout()
plt.show()
