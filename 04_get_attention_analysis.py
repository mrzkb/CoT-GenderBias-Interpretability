import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_heatmap(sas_matrix, title, output_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(sas_matrix, cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Average SAS'})
    plt.xlabel('Head')
    plt.ylabel('Layer')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()  # Close figure to free memory
    print(f"Saved: {output_path}")
    return

# Create output directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

cases = [
    'data/sas_rows_Meta-Llama-3.1-8B-Instruct_prompts_gender_ambig_cot.csv',
    'data/sas_rows_Meta-Llama-3.1-8B-Instruct_prompts_gender_ambig_no_cot.csv',
    'data/sas_rows_Meta-Llama-3.1-8B-Instruct_prompts_gender_disambig_cot.csv',
    'data/sas_rows_Meta-Llama-3.1-8B-Instruct_prompts_gender_disambig_no_cot.csv'
]

# Define readable names for each case
case_names = {
    'gender_ambig_cot': 'Gender Ambiguous with CoT',
    'gender_ambig_no_cot': 'Gender Ambiguous without CoT',
    'gender_disambig_cot': 'Gender Disambiguated with CoT',
    'gender_disambig_no_cot': 'Gender Disambiguated without CoT'
}

for case in cases:
    # Extract case identifier from filename
    if 'gender_ambig_cot' in case:
        case_id = 'gender_ambig_cot'
    elif 'gender_ambig_no_cot' in case:
        case_id = 'gender_ambig_no_cot'
    elif 'gender_disambig_cot' in case:
        case_id = 'gender_disambig_cot'
    elif 'gender_disambig_no_cot' in case:
        case_id = 'gender_disambig_no_cot'
    
    # Load the per-head SAS data
    sas_df = pd.read_csv(case)

    # Create pivot table for heatmap (layers x heads matrix)
    sas_matrix = sas_df.groupby(['layer', 'head'])['sas'].mean().unstack()
    
    # Create descriptive title and filename
    title = f'Average SAS per Head - {case_names[case_id]}'
    output_path = f'figures/sas_heatmap_{case_id}.png'
    
    create_heatmap(sas_matrix, title, output_path)

# normalizing?
# heatmap of the change or the raw SAS?

# Basic Aggregation Across All Prompts

# Separate Prompts by Case

# Heatmap for Cases.