import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

df = pd.read_csv("gender_bbq/results/attention_results/attention_per_prompt_no_COT.csv")

num_layers, num_heads = map(int, df["attn_matrix_shape"].iloc[0].split("x"))


bias_tensor = np.zeros((len(df), num_layers, num_heads))
anti_bias_tensor = np.zeros((len(df), num_layers, num_heads))


for i, row in df.iterrows():
    bias_flat = np.array(list(map(float, row["bias_matrix_flat"].split(","))))
    anti_flat = np.array(list(map(float, row["anti_bias_matrix_flat"].split(","))))
    bias_tensor[i] = bias_flat.reshape((num_layers, num_heads))
    anti_bias_tensor[i] = anti_flat.reshape((num_layers, num_heads))

diff_tensor = bias_tensor - anti_bias_tensor
avg_diff = diff_tensor.mean(axis=0)

plt.figure(figsize=(14, 8))
ax = sns.heatmap(avg_diff, cmap="coolwarm", center=0, vmin=-0.1, vmax=0.1,
                 cbar_kws={'label': 'Attention Bias Score' })


highlight_heads = [(5, 5), (14, 19), (4, 16), (10, 2), (1, 23),
                   (10, 1), (13, 18), (7, 25), (13, 9), (14, 17)]

for layer, head in highlight_heads:
    rect = patches.Rectangle((head, layer), 1, 1, fill=False, edgecolor='red', linewidth=1.5)
    ax.add_patch(rect)

plt.xlabel("Head", fontsize=18)
plt.ylabel("Layer", fontsize=18)

cbar = ax.collections[0].colorbar
cbar.set_label("Attention Bias Score", fontsize=18)

plt.tight_layout()
plt.savefig("attention_diff_per_headxlayer_no_COT_heatmap.png", dpi=300)
plt.show()
