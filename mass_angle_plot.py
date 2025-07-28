import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === CONFIGURATION ===
CSV_FOLDER = '.'  # Current folder
USE_STD = True
pressure_order = ['50bar', '60bar', '70bar']
Y_LIMS = (4, 18)
Y_TICKS = np.arange(Y_LIMS[0], Y_LIMS[1] + 1, 2)
OUTPUT_CSV = 'spray_angle_summary.csv'
# =====================

# === MAPPINGS ===
FUEL_MAP = {'etanol': 'Ethanol', 'gasolina': 'Gasoline'}
NOZZLE_MAP = {'conv': 'Convergent', 'div': 'Divergent'}
FUEL_COLOR = {'Ethanol': 'red', 'Gasoline': 'blue'}
TEMP_MARKER = {'25 °C': 'o', '40 °C': 's'}

# === EXTRACTORS ===
def extract_fuel(csv_name):
    match = re.search(r'spray_angles_(etanol|gasolina)_', csv_name)
    return FUEL_MAP.get(match.group(1), match.group(1).capitalize()) if match else 'Unknown'

def extract_temperature(csv_name):
    match = re.search(r'_(\d{2}C)_\d+bar', csv_name)
    return match.group(1).replace('C', ' °C') if match else 'Unknown'

def extract_pressure(csv_name):
    match = re.search(r'_(\d+)bar', csv_name)
    return match.group(1) + 'bar' if match else 'Unknown'

def extract_nozzle(csv_name):
    match = re.search(r'_(conv|div)_\d{2}C', csv_name)
    return NOZZLE_MAP.get(match.group(1), match.group(1).capitalize()) if match else 'Unknown'

# === LOAD DATA ===
csv_files = [f for f in os.listdir(CSV_FOLDER)
             if f.lower().endswith('.csv') and f.startswith('spray_angles_')]

data = {}  # (fuel, nozzle, temp) → pressure → (mean, std)
rows = []

for file in sorted(csv_files):
    try:
        df = pd.read_csv(os.path.join(CSV_FOLDER, file))
        df = df[pd.to_numeric(df['Angle (degrees)'], errors='coerce').notnull()]
        angles = df['Angle (degrees)'].astype(float)

        mean = angles.mean()
        error = angles.std() if USE_STD else (angles - mean).abs().mean()

        fuel = extract_fuel(file)
        nozzle = extract_nozzle(file)
        temp = extract_temperature(file)
        pressure = extract_pressure(file)

        key = (fuel, nozzle, temp)
        if key not in data:
            data[key] = {}
        data[key][pressure] = (mean, error)

        rows.append({
            'Fuel': fuel,
            'Nozzle': nozzle,
            'Temperature': temp,
            'Pressure': pressure,
            'Mean Angle (deg)': round(mean, 2),
            'Deviation': round(error, 2)
        })

    except Exception as e:
        print(f"[!] Failed to process {file}: {e}")

# === SAVE SUMMARY CSV ===
summary_df = pd.DataFrame(rows)
summary_df = summary_df.sort_values(by=['Fuel', 'Nozzle', 'Temperature', 'Pressure'])
summary_df.to_csv(OUTPUT_CSV, index=False)

# === PRINT SUMMARY TO TERMINAL ===
print("\n=== Spray Angle Summary ===")
print(summary_df.to_string(index=False))

# === PLOT FOR EACH NOZZLE TYPE ===
for nozzle_type in ['Convergent', 'Divergent']:
    fig, ax = plt.subplots(figsize=(10, 6))

    for (fuel, nozzle, temp), pressures_dict in data.items():
        if nozzle != nozzle_type:
            continue

        means = []
        errors = []
        xs = []

        for i, pressure in enumerate(pressure_order):
            if pressure in pressures_dict:
                m, e = pressures_dict[pressure]
                means.append(m)
                errors.append(e)
                xs.append(i)

        if means:
            label = f"{fuel}, {temp}"
            color = FUEL_COLOR.get(fuel, 'black')
            marker = TEMP_MARKER.get(temp, 'o')

            ax.errorbar(
                xs, means, yerr=errors,
                fmt=marker + '--',
                label=label,
                color=color,
                capsize=5,
                markersize=8
            )

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['50', '60', '70'], fontsize=12)
    ax.set_yticks(Y_TICKS)
    ax.set_ylim(Y_LIMS)
    ax.set_xlabel('Pressure [bar]', fontweight='bold', fontsize=13)
    ax.set_ylabel('Angle [°]', fontweight='bold', fontsize=13)
    ax.set_title(f'Angle × Pressure – {nozzle_type} Nozzle', fontweight='bold', fontsize=14)
    ax.tick_params(axis='y', labelsize=13, labelcolor='black', width=1.5)
    ax.tick_params(axis='x', labelsize=13, labelcolor='black', width=1.5)
    ax.grid(True, axis='y', linestyle='--')

    legend = ax.legend(
        title='Fuel and Gas Phase Temperature',
        title_fontproperties={'weight': 'bold', 'size': 13},
        fontsize=13
    )

    fig.tight_layout()
    filename = f'Spray_angle_variance_{nozzle_type.lower()}.png'
    fig.savefig(filename)
    plt.show()
