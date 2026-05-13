import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === CONFIGURAÇÃO DE CAMINHOS ===
ROOT_DIR = "/run/media/mglopes/MGL_SSD/HIGHSPEEDIMAG/"
CSV_PATH = os.path.join(ROOT_DIR, "Spray_Angles", "Spray_Angles_Report.csv")
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "Spray_Angles")

# === MAPEAMENTO DE ESTILO (Foco em Acessibilidade e Padronização) ===
# Cor única de alto contraste; Marcadores distintos para cada caso.
MAIN_COLOR = "#002366"  # Royal Blue escuro

CASE_STYLE = {
    'agua35':       {'label': 'Distilled Water @ 35°C',      'marker': 'o'}, # Círculo
    'agua_45':      {'label': 'Distilled Water @ 45°C',      'marker': 's'}, # Quadrado
    'etanol75_35':  {'label': 'Hydrous Ethanol 75% @ 35°C', 'marker': '^'}, # Triângulo
    'etanol75_45':  {'label': 'Hydrous Ethanol 75% @ 45°C', 'marker': 'D'}, # Diamante
    'etanol95_35':  {'label': 'Hydrous Ethanol 95% @ 35°C', 'marker': 'v'}, # Triângulo invertido
    'etanol95_45':  {'label': 'Hydrous Ethanol 95% @ 45°C', 'marker': 'P'}, # Cruz preenchida
}

def format_case_string(raw_case):
    """Extrai a parte útil após o espaço."""
    return raw_case.split(' ')[-1].strip()

# === CARREGAMENTO ===
if not os.path.exists(CSV_PATH):
    print(f"[!] Arquivo não encontrado em: {CSV_PATH}")
else:
    df = pd.read_csv(CSV_PATH)
    df['Clean_Key'] = df['Case'].apply(format_case_string)
    
    # Ordenação Alfabética (agua -> etanol75 -> etanol95)
    df = df.sort_values('Clean_Key').reset_index(drop=True)

    # === PLOTAGEM ===
    fig, ax = plt.subplots(figsize=(11, 7))

    for i, row in df.iterrows():
        key = row['Clean_Key']
        style = CASE_STYLE.get(key, {'label': key, 'marker': 'o'})
        
        ax.scatter(
            i, 
            row['Angle_Degrees'], 
            color=MAIN_COLOR, 
            marker=style['marker'], 
            s=180,           # Marcadores levemente maiores para visibilidade
            label=style['label'],
            edgecolors='black',
            linewidths=1.0,
            alpha=0.9,       # Leve transparência para sobreposição se houver
            zorder=3
        )

    # Configuração do Eixo X
    ax.set_xticks(range(len(df)))
    # Labels curtas no eixo X para evitar poluição visual
    ax.set_xticklabels(df['Clean_Key'], fontsize=11, fontweight='500')

    # Configuração do Eixo Y (Intervalos de 2 graus para precisão)
    y_min = int(df['Angle_Degrees'].min() - 4)
    y_max = int(df['Angle_Degrees'].max() + 4)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(y_min, y_max + 1, 2))
    
    # Grid sutil para acessibilidade (ajuda a guiar o olhar até o valor do eixo Y)
    ax.grid(True, linestyle='--', alpha=0.4, color='gray', zorder=0)
    
    # Títulos e Labels
    ax.set_title('Comparative Analysis of Spray Plume Angles', fontweight='bold', fontsize=15, pad=25)
    ax.set_ylabel('Plume Angle [°]', fontweight='bold', fontsize=13)
    ax.set_xlabel('Experimental Conditions (Short ID)', fontweight='bold', fontsize=13)

    # Legenda com o título solicitado
    ax.legend(
        bbox_to_anchor=(1.02, 1), 
        loc='upper left', 
        title="Tested Cases", 
        title_fontproperties={'weight':'bold', 'size': 12},
        frameon=True,
        shadow=False,
        fontsize=11
    )

    plt.tight_layout()
    
    # Salvar em alta resolução
    save_path = os.path.join(OUTPUT_FOLDER, "Comparison_Spray_Angles_Standardized.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"\n[OK] Gráfico padronizado salvo em: {save_path}")
    plt.show()