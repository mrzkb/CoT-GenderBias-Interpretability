import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

no_cot_df = pd.read_csv("gender_bbq/results/attention_results/attention_per_prompt_no_COT.csv")
cot_df = pd.read_csv("gender_bbq/results/attention_results/attention_per_prompt_COT.csv")

num_layers, num_heads = map(int, no_cot_df["attn_matrix_shape"].iloc[0].split("x"))

merged_df = pd.merge(no_cot_df, cot_df, on="prompt_index", suffixes=("_no_cot", "_cot"))

category_map = {
    ("correct", "correct"): "1_correct_to_correct",
    ("correct", "bias_wrong"): "2_correct_to_bias",
    ("correct", "anti_bias_wrong"): "3_correct_to_anti_bias",
    ("bias_wrong", "correct"): "4_bias_to_correct",
    ("bias_wrong", "bias_wrong"): "5_bias_to_bias",
    ("bias_wrong", "anti_bias_wrong"): "6_bias_to_anti_bias",
    ("anti_bias_wrong", "bias_wrong"): "7_anti_bias_to_bias",
    ("anti_bias_wrong", "correct"): "8_anti_bias_to_correct",
    ("anti_bias_wrong", "anti_bias_wrong"): "9_anti_bias_to_anti_bias"
}
category_deltas = {k: [] for k in category_map.values()}
category_deltas["10_all"] = []

for _, row in merged_df.iterrows():
    key = (row["prompt_group_no_cot"], row["prompt_group_cot"])
    category = category_map.get(key, None)

    bias_no = np.fromstring(row["bias_matrix_flat_no_cot"], sep=",").reshape(num_layers, num_heads)
    anti_no = np.fromstring(row["anti_bias_matrix_flat_no_cot"], sep=",").reshape(num_layers, num_heads)
    bias_cot = np.fromstring(row["bias_matrix_flat_cot"], sep=",").reshape(num_layers, num_heads)
    anti_cot = np.fromstring(row["anti_bias_matrix_flat_cot"], sep=",").reshape(num_layers, num_heads)

    delta = (bias_cot - anti_cot) - (bias_no - anti_no)
    if category:
        category_deltas[category].append(delta)
    category_deltas["10_all"].append(delta)

output_dir = "gender_bbq/results/attention_results/category_deltas/global_style"
os.makedirs(output_dir, exist_ok=True)

def save_outputs(label, delta_list):
    if not delta_list:
        return
    stacked = np.stack(delta_list)
    avg_delta = stacked.mean(axis=0)

    np.savetxt(os.path.join(output_dir, f"avg_delta_{label}.csv"), avg_delta, delimiter=",")

    plt.figure(figsize=(14, 8))
    ax = sns.heatmap(avg_delta, cmap="coolwarm", center=0, vmin=-0.1, vmax=0.1,
                     cbar_kws={'label': 'Δ (Stereotypical – Anti-Stereotypical Attention, CoT – NoCoT)'})
    plt.xlabel("Head")
    plt.ylabel("Layer")
    ax.text(num_heads / 2, -1.5, label.replace('_', ' ').upper(), fontsize=18, fontweight='bold', ha='center')
    plt.title(f"Number of Prompts: {len(delta_list)}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"heatmap_{label}.png"), dpi=300)
    plt.close()

for label, delta_list in category_deltas.items():
    save_outputs(label, delta_list)

all_deltas = category_deltas["10_all"]

if all_deltas:
    stacked = np.stack(all_deltas)
    prompt_means = stacked.mean(axis=(1, 2))

    inc = np.sum(prompt_means > 0)
    dec = np.sum(prompt_means < 0)
    zero = np.sum(prompt_means == 0)
    total = len(prompt_means)

    pct_inc = 100 * inc / total
    pct_dec = 100 * dec / total
    pct_zero = 100 * zero / total

    summary_df = pd.DataFrame([{
        "category": "10_all",
        "% increased": round(pct_inc, 2),
        "% decreased": round(pct_dec, 2),
        "% no change": round(pct_zero, 2),
        "num prompts": total
    }])

    summary_df.to_csv(os.path.join(output_dir, "global_delta_summary.csv"), index=False)

if all_deltas:
    stacked_deltas = np.stack(all_deltas)
    stacked_bias_no = np.stack([
        np.fromstring(row["bias_matrix_flat_no_cot"], sep=",").reshape(num_layers, num_heads)
        for _, row in merged_df.iterrows()
    ])
    stacked_anti_no = np.stack([
        np.fromstring(row["anti_bias_matrix_flat_no_cot"], sep=",").reshape(num_layers, num_heads)
        for _, row in merged_df.iterrows()
    ])
    stacked_bias_cot = np.stack([
        np.fromstring(row["bias_matrix_flat_cot"], sep=",").reshape(num_layers, num_heads)
        for _, row in merged_df.iterrows()
    ])
    stacked_anti_cot = np.stack([
        np.fromstring(row["anti_bias_matrix_flat_cot"], sep=",").reshape(num_layers, num_heads)
        for _, row in merged_df.iterrows()
    ])

    avg_gap_no = (stacked_bias_no - stacked_anti_no).mean(axis=0)
    avg_gap_cot = (stacked_bias_cot - stacked_anti_cot).mean(axis=0)
    avg_change = avg_gap_cot - avg_gap_no

    flat_change = avg_change.flatten()
    top_indices = np.argsort(np.abs(flat_change))[-10:][::-1]  # top 10 largest absolute changes
    rows = []

    for idx in top_indices:
        layer, head = divmod(idx, num_heads)
        delta_before = avg_gap_no[layer, head]
        delta_after = avg_gap_cot[layer, head]
        delta_change = delta_after - delta_before
        direction = "increase" if delta_change > 0 else "decrease"

        rows.append({
            "Layer": layer,
            "Head": head,
            "Δ before (No-CoT)": round(delta_before, 4),
            "Δ after (CoT)": round(delta_after, 4),
            "Change in Δ": round(delta_change, 4),
            "Direction": direction
        })

    top_heads_df = pd.DataFrame(rows)
    top_heads_df.to_csv(os.path.join(output_dir, "top10_heads_by_delta_change.csv"), index=False)

