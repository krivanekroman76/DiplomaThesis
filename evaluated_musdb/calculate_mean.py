import os
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# Paths
# =========================

EVAL_ROOT = Path("evaluated_musdb")
INPUT_JSON = EVAL_ROOT / "museval_results.json"
OUTPUT_MEAN_JSON = EVAL_ROOT / "museval_tool_means.json"
OUTPUT_LATEX_TABLE = EVAL_ROOT / "museval_tool_means_table.txt"
OUTPUT_PLOT = EVAL_ROOT / "separation_metrics.png"

# =========================
# Helper functions
# =========================

def latex_escape(text: str) -> str:
    """Escape LaTeX special characters."""
    return text.replace("_", "\\_")

# =========================
# Load results
# =========================

with open(INPUT_JSON, "r") as f:
    results = json.load(f)

# =========================
# Aggregate means per tool/model
# =========================

def split_system(system_str):
    """Split full system into tool and model (Spleeter has no model)"""
    if "_" in system_str:
        tool, model = system_str.split("_", 1)
    else:
        tool, model = system_str, ""
    return tool, model

grouped = defaultdict(list)
for r in results:
    key = (r["system"], r["target"])
    grouped[key].append(r)

tool_means = []

for (system, target), items in grouped.items():
    mean_sdr = np.mean([np.mean(i["SDR"]) for i in items])
    mean_sir = np.mean([np.mean(i["SIR"]) for i in items])
    mean_sar = np.mean([np.mean(i["SAR"]) for i in items])
    mean_isr = np.mean([np.mean(i["ISR"]) for i in items])
    
    tool_means.append({
        "system": system,
        "tool": split_system(system)[0],
        "model": split_system(system)[1],
        "target": target,
        "mean_SDR": mean_sdr,
        "mean_SIR": mean_sir,
        "mean_SAR": mean_sar,
        "mean_ISR": mean_isr
    })

# Save JSON
with open(OUTPUT_MEAN_JSON, "w") as f:
    json.dump(tool_means, f, indent=2)
print(f"[INFO] Mean values per tool saved to {OUTPUT_MEAN_JSON}")

# =========================
# Create Order for table and plot
# =========================

TOOL_ORDER = {
    "Spleeter": 0,
    "Demucs": 1,
    "OpenUnmix": 2
}

MODEL_ORDER = {
    "": 0,            # Spleeter
    "mdx": 1,
    "mdx_extra": 2,
    "htdemucs": 3,
    "umx": 1,
    "umxhq": 2,
    "umxl": 3
}

tool_means.sort(
    key=lambda x: (
        TOOL_ORDER.get(x["tool"], 99),
        MODEL_ORDER.get(x["model"], 99),
        x["target"]
    )
)

# =========================
# Create LaTeX table with table number and caption
# =========================

latex_lines = []
latex_lines.append("\\begin{table}[h!]")  # start table environment
latex_lines.append("\\centering")
latex_lines.append("\\caption{Mean separation metrics for each tool/model}")  # table caption
latex_lines.append("\\label{tab:separation_metrics}")  # optional label for referencing
latex_lines.append("\\begin{tabular}{lllcccc}")
latex_lines.append("\\hline")
latex_lines.append("Tool & Model & Target & SDR [dB] & SIR [dB] & SAR [dB] & ISR [dB] \\\\")
latex_lines.append("\\hline")

for item in tool_means:
    line = (
    f"{latex_escape(item['tool'])} & "
    f"{latex_escape(item['model'])} & "
    f"{latex_escape(item['target'])} & "
    f"{item['mean_SDR']:.2f} & {item['mean_SIR']:.2f} & "
    f"{item['mean_SAR']:.2f} & {item['mean_ISR']:.2f} \\\\"
)
    latex_lines.append(line)

latex_lines.append("\\hline")
latex_lines.append("\\end{tabular}")
latex_lines.append("\\end{table}")  # end table environment

with open(OUTPUT_LATEX_TABLE, "w") as f:
    f.write("\n".join(latex_lines))

print(f"[INFO] LaTeX table with caption saved to {OUTPUT_LATEX_TABLE}")

# =========================
# Plot setup: all in one figure with same scale
# =========================
MODEL_COLORS = {
    "": "#636EFA",           # Spleeter
    "mdx": "#EF553B",
    "mdx_extra": "#AB63FA",
    "htdemucs": "#FFA15A",
    "umx": "#19D3F3",
    "umxhq": "#00CC96",
    "umxl": "#B6E880",
}

fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=["SDR (vocals)", "SDR (accompaniment)"]
)

targets = ["vocals", "accompaniment"] 
metrics = ["mean_SDR", "mean_SIR", "mean_SAR", "mean_ISR"] # Global y-axis scale 
all_values = [item[m] for item in tool_means for m in metrics] 
ymin, ymax = min(all_values), max(all_values) 
row_map = {"vocals": 1, "accompaniment": 2}

for target in targets:
    target_items = [i for i in tool_means if i["target"] == target]
    labels = [f"{i['tool']} {i['model']}".strip() for i in target_items]
    values = [i["mean_SDR"] for i in target_items]

    for i, item in enumerate(target_items):
        fig.add_trace(
            go.Bar(
                x=[labels[i]],
                y=[values[i]],
                marker_color=MODEL_COLORS.get(item["model"], "#888888"),
                showlegend=(target == "vocals"),
                name=f"{item['tool']} {item['model']}".strip()
            ),
            row=row_map[target],
            col=1
        )

fig.update_layout(
    showlegend=False,
    height=600,
    width=800,
    yaxis_title="dB",
    font=dict(size=14)
)

# Replace file if exists
if OUTPUT_PLOT.exists():
    OUTPUT_PLOT.unlink()

fig.write_image(str(OUTPUT_PLOT), scale=2)
print(f"[INFO] Separation metrics plot saved to {OUTPUT_PLOT}")


