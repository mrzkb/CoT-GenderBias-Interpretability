'''
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, rankdata
import matplotlib.pyplot as plt
import seaborn as sns

# === Load data ===
df_cot = pd.read_csv("gender_bbq/results/attention_results/attention_per_prompt_COT.csv")
df_nocot = pd.read_csv("gender_bbq/results/attention_results/attention_per_prompt_no_COT.csv")
df = pd.merge(df_cot, df_nocot, on="prompt_index", suffixes=("_cot", "_nocot"))

# Parse matrix shape
num_layers, num_heads = map(int, df["attn_matrix_shape_cot"].iloc[0].split("x"))
total_heads = num_layers * num_heads

# Helper to parse flat attention
def parse_flat(col): return np.array(list(map(float, col.split(","))))

# Parse attention matrices
bias_cot = np.stack(df["bias_matrix_flat_cot"].apply(parse_flat))
#anti_cot = np.stack(df["anti_bias_matrix_flat_cot"].apply(parse_flat))
bias_nocot = np.stack(df["bias_matrix_flat_nocot"].apply(parse_flat))
#anti_nocot = np.stack(df["anti_bias_matrix_flat_nocot"].apply(parse_flat))

# Compute delta (CoT - NoCoT)
#gap_cot = bias_cot - anti_cot
#gap_nocot = bias_nocot - anti_nocot
delta = bias_cot - bias_nocot  # shape: (num_prompts, total_heads)

# Initialize matrices
w_signed_matrix = np.full((num_layers, num_heads), np.nan)
p_matrix = np.full((num_layers, num_heads), np.nan)

# Compute Wilcoxon signed-rank test and signed W
for l in range(num_layers):
    for h in range(num_heads):
        idx = l * num_heads + h
        values = delta[:, idx]
        non_zero = values[values != 0]
        if len(non_zero) < 5:
            continue
        ranks = rankdata(np.abs(non_zero))
        signs = np.sign(non_zero)
        w_signed = np.sum(signs * ranks)
        w_signed_matrix[l, h] = w_signed
        try:
            _, p = wilcoxon(values)
            p_matrix[l, h] = p
        except ValueError:
            continue

# === Heatmap: Signed Wilcoxon W ===
plt.figure(figsize=(14, 8))
sns.heatmap(w_signed_matrix, cmap="coolwarm", center=0,
            cbar_kws={'label': 'Signed Wilcoxon W (ΔBias Attention)'})
plt.xlabel("Head")
plt.ylabel("Layer")
plt.title("Signed Wilcoxon Statistic per Head\nΔ(Bias – AntiBias Attention, CoT – NoCoT)")
plt.tight_layout()
plt.savefig("signed_wilcoxon_statistic_heatmap_fixed.png", dpi=300)
plt.show()

# === Heatmap: Wilcoxon P-values ===
plt.figure(figsize=(14, 8))
sns.heatmap(p_matrix, cmap="coolwarm", vmin=0, vmax=0.1,
            cbar_kws={'label': 'Wilcoxon P-Value'})
plt.xlabel("Head")
plt.ylabel("Layer")
plt.title("Wilcoxon P-values per Head\nΔ(Bias – AntiBias Attention, CoT – NoCoT)")
plt.tight_layout()
plt.savefig("wilcoxon_pvalues_heatmap.png", dpi=300)
plt.show()


import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

# === Load data ===
df_cot = pd.read_csv("gender_bbq/results/attention_results/attention_per_prompt_COT.csv")
df_nocot = pd.read_csv("gender_bbq/results/attention_results/attention_per_prompt_no_COT.csv")
df = pd.merge(df_cot, df_nocot, on="prompt_index", suffixes=("_cot", "_nocot"))

# === Parse matrix shape ===
num_layers, num_heads = map(int, df["attn_matrix_shape_cot"].iloc[0].split("x"))

# === Helper to parse attention matrix ===
def parse_flat(col): return np.array(list(map(float, col.split(","))))

# === Extract biased-token attention matrices ===
bias_cot = np.stack(df["bias_matrix_flat_cot"].apply(parse_flat))
bias_nocot = np.stack(df["bias_matrix_flat_nocot"].apply(parse_flat))

# === Compute delta (CoT – NoCoT) attention to biased token ===
delta = bias_cot - bias_nocot  # shape: (num_prompts, total_heads)

# === Initialize matrices ===
w_matrix = np.full((num_layers, num_heads), np.nan)
p_matrix = np.full((num_layers, num_heads), np.nan)

# === Run Wilcoxon test per head ===
for l in range(num_layers):
    for h in range(num_heads):
        idx = l * num_heads + h
        values = delta[:, idx]
        try:
            stat, p = wilcoxon(values, alternative='less')
            w_matrix[l, h] = stat  # Default W+ returned by scipy
            p_matrix[l, h] = p
        except ValueError:
            continue  # e.g., if all values are zero

# === Plot: Wilcoxon W Statistic ===
plt.figure(figsize=(14, 8))
sns.heatmap(w_matrix, cmap="coolwarm", center=0,
            cbar_kws={'label': 'Wilcoxon W (Bias Attention Δ)'})
plt.xlabel("Head")
plt.ylabel("Layer")
plt.title("Wilcoxon W Statistic per Head\nΔ(Bias Token Attention, CoT – NoCoT)")
plt.tight_layout()
plt.savefig("wilcoxon_bias_attention_W_heatmap.png", dpi=300)
plt.show()

# === Plot: P-values ===
plt.figure(figsize=(14, 8))
sns.heatmap(p_matrix, cmap="viridis", vmin=0, vmax=0.1,
            cbar_kws={'label': 'Wilcoxon P-Value'})
plt.xlabel("Head")
plt.ylabel("Layer")
plt.title("Wilcoxon P-Values per Head\nΔ(Bias Token Attention, CoT – NoCoT)")
plt.tight_layout()
plt.savefig("wilcoxon_bias_attention_pvalues_heatmap.png", dpi=300)
plt.show()

''' 
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

# === Load data ===
df_cot = pd.read_csv("gender_bbq/results/attention_results/attention_per_prompt_COT.csv")
df_nocot = pd.read_csv("gender_bbq/results/attention_results/attention_per_prompt_no_COT.csv")
df = pd.merge(df_cot, df_nocot, on="prompt_index", suffixes=("_cot", "_nocot"))

# === Parse matrix shape ===
num_layers, num_heads = map(int, df["attn_matrix_shape_cot"].iloc[0].split("x"))
total_heads = num_layers * num_heads

# === Helper to parse attention matrix ===
def parse_flat(col): return np.array(list(map(float, col.split(","))))

# === Parse attention matrices ===
bias_cot = np.stack(df["bias_matrix_flat_cot"].apply(parse_flat))
anti_cot = np.stack(df["anti_bias_matrix_flat_cot"].apply(parse_flat))
bias_nocot = np.stack(df["bias_matrix_flat_nocot"].apply(parse_flat))
anti_nocot = np.stack(df["anti_bias_matrix_flat_nocot"].apply(parse_flat))

# === Compute ΔGap: (bias - antibias) under CoT vs NoCoT ===
gap_cot = bias_cot - anti_cot
gap_nocot = bias_nocot - anti_nocot
delta = gap_cot - gap_nocot  # shape: (num_prompts, total_heads)

# === Initialize result matrices ===
w_matrix = np.full((num_layers, num_heads), np.nan)
p_matrix = np.full((num_layers, num_heads), np.nan)

# === Wilcoxon test per head ===
for l in range(num_layers):
    for h in range(num_heads):
        idx = l * num_heads + h
        values = delta[:, idx]
        try:
            stat, p = wilcoxon(values)
            w_matrix[l, h] = stat
            p_matrix[l, h] = p
        except ValueError:
            continue

# === Save p-values & statistics ===
np.savetxt("wilcoxon_gap_statistic.csv", w_matrix, delimiter=",")
np.savetxt("wilcoxon_gap_pvalues.csv", p_matrix, delimiter=",")

# === Heatmap: Wilcoxon W Statistic ===
plt.figure(figsize=(14, 8))
sns.heatmap(w_matrix, cmap="coolwarm", center=0,
            cbar_kws={'label': 'Wilcoxon W (ΔGap: Bias – AntiBias)'})
plt.xlabel("Head")
plt.ylabel("Layer")
plt.title("Wilcoxon W per Head\nΔ(Bias – AntiBias Attention, CoT – NoCoT)")
plt.tight_layout()
plt.savefig("wilcoxon_gap_wstat_heatmap.png", dpi=300)
plt.show()

# === Heatmap: P-Values ===
plt.figure(figsize=(14, 8))
sns.heatmap(p_matrix, cmap="viridis", vmin=0, vmax=0.1,
            cbar_kws={'label': 'Wilcoxon P-Value'})
plt.xlabel("Head")
plt.ylabel("Layer")
plt.title("Wilcoxon P-Values per Head\nΔ(Bias – AntiBias Attention, CoT – NoCoT)")
plt.tight_layout()
plt.savefig("wilcoxon_gap_pvalues_heatmap.png", dpi=300)
plt.show()

# === Optional: Top 20 Heads & Histogram of ΔGap ===
mean_delta = delta.mean(axis=0)
top20_idx = np.argsort(np.abs(mean_delta))[-10:]

# Plot histogram of the ΔGap values for the top 20 heads
top20_deltas = delta[:, top20_idx].flatten()

plt.figure(figsize=(10, 6))
sns.histplot(top20_deltas, bins=50, kde=True, color="darkorange", edgecolor="black")
plt.axvline(0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Δ(Bias – AntiBias Attention) (CoT – NoCoT)")
plt.ylabel("Frequency")
plt.title("Distribution of Δ(Bias – AntiBias Attention)\n(Top 20 Heads by Magnitude of Shift)")
plt.tight_layout()
plt.savefig("top20_gap_shift_histogram.png", dpi=300)
plt.show()

import matplotlib.patches as patches

# === Compute top 20 heads by ΔGap magnitude ===
mean_delta = delta.mean(axis=0)
top20_idx = np.argsort(np.abs(mean_delta))[-10:]
top20_coords = [divmod(i, num_heads) for i in top20_idx]

# === Heatmap with top 20 heads outlined ===
plt.figure(figsize=(14, 8))
ax = sns.heatmap(p_matrix, cmap="viridis", vmin=0, vmax=0.1,
                 cbar_kws={'label': 'Wilcoxon P-Value'})

# Draw rectangles around top 20 heads
for (l, h) in top20_coords:
    rect = patches.Rectangle((h, l), 1, 1, fill=False, edgecolor='red', linewidth=1.5)
    ax.add_patch(rect)

plt.xlabel("Head")
plt.ylabel("Layer")
plt.title("Wilcoxon P-values per Head\nΔ(Bias – AntiBias Attention, CoT – NoCoT)\n(Top 20 Heads Highlighted)")
plt.tight_layout()
plt.savefig("wilcoxon_pvalues_heatmap_top10.png", dpi=300)
plt.show()

# === Compute and save top 10 heads by |mean ΔGap| ===
mean_delta = delta.mean(axis=0)
top10_idx = np.argsort(np.abs(mean_delta))[-10:]  # indices of top 10 heads
top10_coords = [divmod(i, num_heads) for i in top10_idx]  # (layer, head)

# Collect data for export
top10_data = []
for idx, (layer, head) in zip(top10_idx, top10_coords):
    top10_data.append({
        "Layer": layer,
        "Head": head,
        "Mean_DeltaGap": mean_delta[idx],
        "P_Value": p_matrix[layer, head]
    })

# Save as CSV
top10_df = pd.DataFrame(top10_data)
top10_df = top10_df.sort_values("Mean_DeltaGap", key=np.abs, ascending=False)
top10_df.to_csv("top10_heads_deltaGap.csv", index=False)

print("Top 10 heads saved to top10_heads_deltaGap.csv")
