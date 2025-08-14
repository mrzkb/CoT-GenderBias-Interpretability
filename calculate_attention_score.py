import pandas as pd
import numpy as np

df = pd.read_csv("gender_bbq/results/attention_results/attention_per_prompt_no_COT.csv")

def parse_matrix(matrix_str, shape_str):
    flat = list(map(float, matrix_str.split(",")))
    rows, cols = map(int, shape_str.split("x"))
    return np.array(flat).reshape((rows, cols))

output_rows = []

for idx, row in df.iterrows():
    try:
        attn_matrix_1 = parse_matrix(row["bias_matrix_flat"], row["attn_matrix_shape"])
        attn_matrix_2 = parse_matrix(row["anti_bias_matrix_flat"], row["attn_matrix_shape"])

        avg_attn_1 = attn_matrix_1.mean(axis=1)
        avg_attn_2 = attn_matrix_2.mean(axis=1)

        attn_diff = np.abs(avg_attn_1 - avg_attn_2)

        output_rows.append({
            "prompt_index": row["prompt_index"],
            "sensitive_word_1": row["sensitive_word_1"],
            "sensitive_word_2": row["sensitive_word_2"],
            "avg_attn_to_token_1": ",".join(map(str, avg_attn_1)),
            "avg_attn_to_token_2": ",".join(map(str, avg_attn_2)),
            "avg_attn_difference": ",".join(map(str, attn_diff))
        })

    except Exception as e:
        print(f"[ERROR] Skipping row {idx}: {e}")

df_out = pd.DataFrame(output_rows)
df_out.to_csv("gender_bbq/results/attention_results/attention_per_layer_no_COT.csv", index=False)