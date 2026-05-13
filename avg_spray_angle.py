import cv2
import numpy as np
import math
import os
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import to_hex
from pathlib import Path

# ==========================================
# 1. CONFIGURAÇÕES GERAIS
# ==========================================
ROOT_DIR = "/run/media/mglopes/MGL_SSD/HIGHSPEEDIMAG/"
INPUT_DIR = os.path.join(ROOT_DIR, "averaged results")
OUTPUT_DIR = os.path.join(ROOT_DIR, "Spray_Angles")

HEIGHT_PIXEL = 50       
THRESHOLD_VALUE = 5     
MAX_DISTANCE = 30        

pivot_point = None

# ==========================================
# 2. FUNÇÕES DE PROCESSAMENTO E FILTRO
# ==========================================

def filter_main_spray(binary_image, max_dist):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    if num_labels < 2: return binary_image
    main_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    main_mask = (labels == main_label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(main_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return main_mask
    contour_mask = np.zeros_like(main_mask)
    cv2.drawContours(contour_mask, contours, -1, 255, 1)
    dist_transform = cv2.distanceTransform(cv2.bitwise_not(contour_mask), cv2.DIST_L2, 5)
    filtered_mask = main_mask.copy()
    for label in range(1, num_labels):
        if label == main_label: continue
        component_mask = (labels == label).astype(np.uint8) * 255
        if np.min(dist_transform[component_mask == 255]) <= max_dist:
            filtered_mask = cv2.bitwise_or(filtered_mask, component_mask)
    return filtered_mask

def select_pivot_point(img_path):
    global pivot_point
    plt.ioff()
    img = cv2.imread(img_path)
    if img is None: return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.canvas.manager.set_window_title('Seletor de Pivô')
    ax.imshow(img_rgb)
    ax.set_title("Selecione o ponto de injeção e feche a janela", pad=20)
    def onclick(event):
        global pivot_point
        if event.xdata is not None and event.ydata is not None:
            pivot_point = (int(event.xdata), int(event.ydata))
            for artist in ax.collections: artist.remove()
            ax.scatter(pivot_point[0], pivot_point[1], color='red', s=60, edgecolors='white')
            fig.canvas.draw()
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

def save_angle_plot(img, pivot, pt_left, pt_right, output_path, case_name, mode, mm_x, mm_y):
    """Gera o gráfico com Grid, sem texto interno, estilo Frequency Map."""
    h, w = img.shape[:2]
    x_max = w * mm_x if mode == "mm" else w
    y_max = h * mm_y if mode == "mm" else h
    
    # Converter pontos para escala (se mm)
    p_x, p_y = (pivot[0]*mm_x, pivot[1]*mm_y) if mode == "mm" else pivot
    l_x, l_y = (pt_left[0]*mm_x, pt_left[1]*mm_y) if mode == "mm" else pt_left
    r_x, r_y = (pt_right[0]*mm_x, pt_right[1]*mm_y) if mode == "mm" else pt_right

    fig, ax = plt.subplots(figsize=(10, 8))
    extent = [0, x_max, y_max, 0]
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), extent=extent, aspect='equal')
    
    # Desenhar o triângulo de demarcação (Matplotlib)
    # Linha Bico -> Esquerda e Bico -> Direita
    ax.plot([p_x, l_x], [p_y, l_y], color='red', linewidth=2, alpha=0.8)
    ax.plot([p_x, r_x], [p_y, r_y], color='red', linewidth=2, alpha=0.8)
    # Linha de base (verde)
    ax.plot([l_x, r_x], [l_y, r_y], color='lime', linewidth=1, linestyle='--', alpha=0.6)
    # Ponto do bico
    ax.scatter(p_x, p_y, color='cyan', s=30, zorder=5)

    # Estilização do Grid (Estilo Frequency Map)
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_max, 0)
    ax.grid(True, linestyle=':', alpha=0.4, color='white')
    ax.set_title(f"Spray Angle Analysis: {case_name}", pad=15)
    ax.set_xlabel("X (mm)" if mode == "mm" else "X (px)")
    ax.set_ylabel("Y (mm)" if mode == "mm" else "Y (px)")

    plt.savefig(output_path, bbox_inches='tight', dpi=120)
    plt.close()

# ==========================================
# 3. LOOP PRINCIPAL
# ==========================================

def run_analysis():
    print("\n" + "="*45)
    print("      SPRAY ANGLE - MODO GRID & ESCALA")
    print("="*45)
    print("1 - Calibração automática (.txt)")
    print("2 - Inserir mm/px manualmente")
    print("3 - Usar escala original (Pixels)")
    opcao = input("\nEscolha uma opção: ").strip()

    mm_x, mm_y, mode = 1.0, 1.0, "pixels"

    if opcao == "1":
        calib_path = os.path.join(ROOT_DIR, "calibration_factor_XY.txt")
        if os.path.exists(calib_path):
            with open(calib_path, 'r') as f:
                for line in f:
                    if "Fator_X" in line: mm_x = float(line.split(":")[1])
                    if "Fator_Y" in line: mm_y = float(line.split(":")[1])
            mode = "mm"
    elif opcao == "2":
        mm_x = float(input("Fator X (mm/px): ").replace(',','.'))
        mm_y = float(input("Fator Y (mm/px): ").replace(',','.'))
        mode = "mm"

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    images = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.tif'))])
    
    if not images: return

    select_pivot_point(os.path.join(INPUT_DIR, images[0]))
    if pivot_point is None: return

    results = []
    for img_name in images:
        case_name = img_name.replace("mean_", "").split(".")[0]
        img_path = os.path.join(INPUT_DIR, img_name)
        img = cv2.imread(img_path)
        
        # Processamento binário e filtro
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
        clean_binary = filter_main_spray(binary, MAX_DISTANCE)
        
        row = clean_binary[HEIGHT_PIXEL, :]
        white_pixels = np.where(row == 255)[0]
        
        if len(white_pixels) >= 2:
            pt_left = (white_pixels[0], HEIGHT_PIXEL)
            pt_right = (white_pixels[-1], HEIGHT_PIXEL)
            
            # Cálculo do ângulo
            v1 = np.array([pt_left[0] - pivot_point[0], pt_left[1] - pivot_point[1]])
            v2 = np.array([pt_right[0] - pivot_point[0], pt_right[1] - pivot_point[1]])
            angle = math.degrees(math.acos(np.clip(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1.0, 1.0)))
            
            # Salvar imagem com Grid e triângulo (sem texto)
            out_img_path = os.path.join(OUTPUT_DIR, f"Angle_{case_name}.png")
            save_angle_plot(img, pivot_point, pt_left, pt_right, out_img_path, case_name, mode, mm_x, mm_y)
            
            results.append([case_name, round(angle, 2)])
            print(f" [OK] {case_name: <25} | {round(angle, 2):>6}°")

    # Salvar Relatório
    with open(os.path.join(OUTPUT_DIR, "Spray_Angles_Report.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Case', 'Angle_Degrees'])
        writer.writerows(results)

    print("\n>>> PROCESSO FINALIZADO! Verifique a pasta Spray_Angles.")

if __name__ == "__main__":
    run_analysis()