import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import os

no_cot_df = pd.read_csv("gender_bbq/results/attention_results/attention_per_prompt_no_COT.csv")
cot_df = pd.read_csv("gender_bbq/results/attention_results/attention_per_prompt_COT.csv")

num_layers, num_heads = map(int, no_cot_df["attn_matrix_shape"].iloc[0].split("x"))

merged_df = pd.merge(
    no_cot_df,
    cot_df,
    on="prompt_index",
    suffixes=("_no_cot", "_cot")
)

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

broad_category_map = {
    ("correct", "bias_wrong"): "correct_to_wrong",
    ("correct", "anti_bias_wrong"): "correct_to_wrong",
    ("bias_wrong", "bias_wrong"): "wrong_to_wrong",
    ("bias_wrong", "anti_bias_wrong"): "wrong_to_wrong",
    ("anti_bias_wrong", "bias_wrong"): "wrong_to_wrong",
    ("anti_bias_wrong", "anti_bias_wrong"): "wrong_to_wrong",
    ("bias_wrong", "correct"): "wrong_to_correct",
    ("anti_bias_wrong", "correct"): "wrong_to_correct"
}

category_deltas = {name: [] for name in category_map.values()}
category_deltas["10_all"] = []
broad_deltas = {name: [] for name in set(broad_category_map.values())}

for _, row in merged_df.iterrows():
    key = (row["prompt_group_no_cot"], row["prompt_group_cot"])
    category = category_map.get(key)
    broad_category = broad_category_map.get(key)

    # reconstruct attention matrices
    bias_no = np.array(list(map(float, row["bias_matrix_flat_no_cot"].split(",")))).reshape((num_layers, num_heads))
    anti_no = np.array(list(map(float, row["anti_bias_matrix_flat_no_cot"].split(",")))).reshape((num_layers, num_heads))
    bias_cot = np.array(list(map(float, row["bias_matrix_flat_cot"].split(",")))).reshape((num_layers, num_heads))
    anti_cot = np.array(list(map(float, row["anti_bias_matrix_flat_cot"].split(",")))).reshape((num_layers, num_heads))

    delta = (bias_cot - anti_cot) - (bias_no - anti_no)

    if category:
        category_deltas[category].append(delta)
    if broad_category:
        broad_deltas[broad_category].append(delta)
    category_deltas["10_all"].append(delta)

output_dir = "gender_bbq/results/attention_results/category_deltas/attention_change_all_categories/outlined"

os.makedirs(output_dir, exist_ok=True)

def save_outputs(category_label, matrices, prefix="avg_delta_", heatmap_prefix="heatmap_"):
    if matrices:
        stacked = np.stack(matrices)
        avg = stacked.mean(axis=0)
        count = len(matrices)

        np.savetxt(os.path.join(output_dir, f"{prefix}{category_label}.csv"), avg, delimiter=",")

        plt.figure(figsize=(14, 8))
        ax = sns.heatmap(avg, cmap="coolwarm", center=0, vmin=-0.1, vmax=0.1,
                         cbar_kws={'label': 'Δ (Stereotypical – Anti-Stereotypical Attention, CoT – NoCoT)'})

        flat_indices = np.argsort(np.abs(avg.flatten()))[::-1][:10]
        top_coords = [divmod(idx, num_heads) for idx in flat_indices]  # (layer, head)

        for (layer, head) in top_coords:
            rect = patches.Rectangle((head, layer), 1, 1, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

        plt.xlabel("Head", fontsize=18)
        plt.ylabel("Layer", fontsize=18)
        #ax.text(num_heads / 2, -1.5, category_label.replace('_', ' ').upper(),
          #      fontsize=18, fontweight='bold', ha='center', va='center')
        #plt.title(f"Number of Prompts: {count}", fontsize=12)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)  # tick label size
        cbar.set_label("Δ Attention Bias Score", fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('global_change_in_attention_before_after_COT.png', dpi=300)
        plt.close()

# Save fine-grained and broad category outputs
#for cat, matrices in category_deltas.items():
 #   save_outputs(cat, matrices)

#for broad_cat, matrices in broad_deltas.items():
  #  save_outputs(broad_cat, matrices, prefix="avg_delta_broad_", heatmap_prefix="heatmap_broad_")

save_outputs("10_all", category_deltas["10_all"])
