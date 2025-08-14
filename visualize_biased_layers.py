import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("gender_bbq/results/attention_results/attention_per_layer_no_COT.csv")

df["attn_diff_list"] = df["avg_attn_difference"].apply(lambda x: list(map(float, x.split(","))))

layer_df = pd.DataFrame(df["attn_diff_list"].to_list())
layer_df.index = df["prompt_index"]  # Set prompt index as row index
# Normalize each row (prompt) by its max difference across all layers
normalized_df = layer_df.div(layer_df.max(axis=1), axis=0)
normalized_df = normalized_df.fillna(0)


# Optional: Rename columns to indicate layer numbers
layer_df.columns = [f"layer_{i}" for i in range(layer_df.shape[1])]

'''
plt.figure(figsize=(14, 6))

for i in range(layer_df.shape[1]):
    sns.kdeplot(layer_df.iloc[:, i], label=f'Layer {i}', alpha=0.3)

plt.xlabel("Attention Difference")
plt.ylabel("Density")
plt.title("Layer-wise Attention Difference Distribution (No CoT)")
plt.legend(loc="upper right", ncol=2)
plt.tight_layout()
plt.savefig("attention_diff_layerwise_kde_no_COT.png", dpi=300)
plt.show()
'''

# Sum attention differences across layers for each prompt
total_diff_per_prompt = layer_df.sum(axis=1)

'''
# Plot histogram
plt.figure(figsize=(10, 6))
sns.histplot(total_diff_per_prompt, bins=30, kde=True, color="steelblue")
plt.ylim(0, 1000)
plt.xlabel("Sum of Attention Differences (Stereotypical – Anti-Stereotypical)")
plt.ylabel("Number of Prompts")
plt.title("Distribution of Total Attention Differences per Prompt (CoT)")
#plt.xlim(right=1000)  # Set max x-axis value to 1000
plt.tight_layout()
plt.savefig("attention_diff_histogram_COT.png", dpi=300)
plt.show()
'''


# Plot the heatmap
plt.figure(figsize=(14, 8))
ax = sns.heatmap(normalized_df.T, cmap="Greens", yticklabels=True)

# Add axis labels
plt.xlabel("Prompt Index", fontsize=18)
plt.ylabel("Layer", fontsize=18)

# Set label for the colorbar to explain the scale
colorbar = ax.collections[0].colorbar
colorbar.set_label("Normalized Absolute Attention Difference", fontsize=18)

plt.tight_layout()
plt.savefig("attention_difference_heatmap_no_COT.png", dpi=300)
plt.show()
plt.close()


'''
cot_df = pd.read_csv("gender_bbq/results/attention_results/attention_per_layer_COT.csv")
no_cot_df = pd.read_csv("gender_bbq/results/attention_results/attention_per_layer_no_COT.csv")

# Convert from comma-separated to list of floats
cot_df["attn_diff_list"] = cot_df["avg_attn_difference"].apply(lambda x: list(map(float, x.split(","))))
no_cot_df["attn_diff_list"] = no_cot_df["avg_attn_difference"].apply(lambda x: list(map(float, x.split(","))))

# Compute sum across all layers (total attention difference per prompt)
cot_total_diff = pd.Series([sum(x) for x in cot_df["attn_diff_list"]])
no_cot_total_diff = pd.Series([sum(x) for x in no_cot_df["attn_diff_list"]])

# Create KDE plot
plt.figure(figsize=(10, 6))
sns.kdeplot(no_cot_total_diff, fill=True, color="darkorange", label="No CoT", linewidth=2, alpha=0.6)
sns.kdeplot(cot_total_diff, fill=True, color="steelblue", label="CoT", linewidth=2, alpha=0.6)

plt.xlabel("Sum of Attention Differences (Stereotypical – Anti-Stereotypical)")
plt.ylabel("Density")
plt.title("KDE of Total Attention Differences per Prompt (CoT vs No CoT)")
plt.legend(title="Prompt Type")
plt.tight_layout()
plt.savefig("attention_diff_kde_CoT_vs_NoCoT.png", dpi=300)
plt.show()
'''
cot_df = pd.read_csv("gender_bbq/results/attention_results/attention_per_layer_COT.csv")
no_cot_df = pd.read_csv("gender_bbq/results/attention_results/attention_per_layer_no_COT.csv")

# Parse and sum
cot_df["attn_diff_list"] = cot_df["avg_attn_difference"].apply(lambda x: list(map(float, x.split(","))))
no_cot_df["attn_diff_list"] = no_cot_df["avg_attn_difference"].apply(lambda x: list(map(float, x.split(","))))

cot_total_diff = pd.Series([sum(x) for x in cot_df["attn_diff_list"]])
no_cot_total_diff = pd.Series([sum(x) for x in no_cot_df["attn_diff_list"]])

# Plot with count-based y-axis
plt.figure(figsize=(10, 6))
sns.histplot(no_cot_total_diff, bins=50, color="hotpink", label="Before CoT", kde=True, alpha=0.5)
sns.histplot(cot_total_diff, bins=50, color="lightblue", label="After CoT", kde=True, alpha=0.5)

plt.xlabel("Sum of Layer Attention Bias Scores per Prompt", fontsize=18)
plt.ylabel("Prompt Count", fontsize=18)  # <-- This now reflects actual number of prompts
#plt.title("Overlayed Histograms with KDE (Counts) for CoT vs No-CoT")
plt.legend(title="Prompt Type", fontsize=18)
plt.tight_layout()
plt.savefig("overlay_histogram_kde_counts.png", dpi=300)
plt.show()
