import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def create_heatmap(sas_matrix, title, output_path, cbar_label):
    plt.figure(figsize=(12, 10))
    sns.heatmap(sas_matrix, cmap='RdBu_r', center=0, vmin=-5, vmax=5, cbar_kws={'label': cbar_label})
    plt.xlabel('Head')
    plt.ylabel('Layer')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")
    return

# Create output directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Group cases by condition (ambig vs disambig)
case_pairs = [
    {
        'condition': 'gender_ambig',
        'no_cot': {
            'sas': 'data/sas_rows_Meta-Llama-3.1-8B-Instruct_prompts_gender_ambig_no_cot.csv',
            'predictions': 'data/Meta-Llama-3.1-8B-Instruct_responses_gender_ambig_no_cot.csv'
        },
        'cot': {
            'sas': 'data/sas_rows_Meta-Llama-3.1-8B-Instruct_prompts_gender_ambig_cot.csv',
            'predictions': 'data/Meta-Llama-3.1-8B-Instruct_responses_gender_ambig_cot.csv'
        }
    },
    {
        'condition': 'gender_disambig',
        'no_cot': {
            'sas': 'data/sas_rows_Meta-Llama-3.1-8B-Instruct_prompts_gender_disambig_no_cot.csv',
            'predictions': 'data/Meta-Llama-3.1-8B-Instruct_responses_gender_disambig_no_cot.csv'
        },
        'cot': {
            'sas': 'data/sas_rows_Meta-Llama-3.1-8B-Instruct_prompts_gender_disambig_cot.csv',
            'predictions': 'data/Meta-Llama-3.1-8B-Instruct_responses_gender_disambig_cot.csv'
        }
    }
]

# Define readable names
case_names = {
    'gender_ambig_cot': 'Gender Ambiguous with CoT',
    'gender_ambig_no_cot': 'Gender Ambiguous without CoT',
    'gender_disambig_cot': 'Gender Disambiguated with CoT',
    'gender_disambig_no_cot': 'Gender Disambiguated without CoT'
}

for pair in case_pairs:
    condition = pair['condition']
    is_disambig = 'disambig' in condition
    
    print(f"\n{'='*60}")
    print(f"Processing: {condition}")
    print(f"{'='*60}")
    
    # STEP 1: Load no_cot predictions to identify groups
    no_cot_predictions = pd.read_csv(pair['no_cot']['predictions'])
    
    # STEP 2: Identify groups based on no_cot predictions
    if is_disambig:
        # For disambiguated cases: split by correctness AND bias
        unknown_ids = no_cot_predictions[
            no_cot_predictions['predicted_answer'].astype(int) == no_cot_predictions['unknown_label'].astype(int)
        ]['example_id'].unique()
        
        bias_correct_ids = no_cot_predictions[
            (no_cot_predictions['predicted_answer'].astype(int) == no_cot_predictions['bias_label'].astype(int)) &
            (no_cot_predictions['predicted_answer'].astype(int) == no_cot_predictions['answer_label'].astype(int))
        ]['example_id'].unique()
        
        bias_incorrect_ids = no_cot_predictions[
            (no_cot_predictions['predicted_answer'].astype(int) == no_cot_predictions['bias_label'].astype(int)) &
            (no_cot_predictions['predicted_answer'].astype(int) != no_cot_predictions['answer_label'].astype(int))
        ]['example_id'].unique()
        
        anti_correct_ids = no_cot_predictions[
            (no_cot_predictions['predicted_answer'].astype(int) != no_cot_predictions['unknown_label'].astype(int)) &
            (no_cot_predictions['predicted_answer'].astype(int) != no_cot_predictions['bias_label'].astype(int)) &
            (no_cot_predictions['predicted_answer'].astype(int) == no_cot_predictions['answer_label'].astype(int))
        ]['example_id'].unique()
        
        anti_incorrect_ids = no_cot_predictions[
            (no_cot_predictions['predicted_answer'].astype(int) != no_cot_predictions['unknown_label'].astype(int)) &
            (no_cot_predictions['predicted_answer'].astype(int) != no_cot_predictions['bias_label'].astype(int)) &
            (no_cot_predictions['predicted_answer'].astype(int) != no_cot_predictions['answer_label'].astype(int))
        ]['example_id'].unique()
        
        print(f"  Unknown label predictions: {len(unknown_ids)} examples")
        print(f"  Bias-correct predictions: {len(bias_correct_ids)} examples")
        print(f"  Bias-incorrect predictions: {len(bias_incorrect_ids)} examples")
        print(f"  Anti-stereotypical-correct predictions: {len(anti_correct_ids)} examples")
        print(f"  Anti-stereotypical-incorrect predictions: {len(anti_incorrect_ids)} examples")
        
        groups = {
            'unknown': (unknown_ids, 'Predicted Unknown Label'),
            'bias_correct': (bias_correct_ids, 'Predicted Bias Label (Correct)'),
            'bias_incorrect': (bias_incorrect_ids, 'Predicted Bias Label (Incorrect)'),
            'anti_correct': (anti_correct_ids, 'Predicted Anti-Stereotypical (Correct)'),
            'anti_incorrect': (anti_incorrect_ids, 'Predicted Anti-Stereotypical (Incorrect)')
        }
        
    else:
        # For ambiguous cases
        unknown_ids = no_cot_predictions[
            no_cot_predictions['predicted_answer'] == no_cot_predictions['unknown_label']
        ]['example_id'].unique()
    
        bias_ids = no_cot_predictions[
            no_cot_predictions['predicted_answer'] == no_cot_predictions['bias_label']
        ]['example_id'].unique()
    
        anti_stereo_ids = no_cot_predictions[
            (no_cot_predictions['predicted_answer'] != no_cot_predictions['unknown_label']) &
            (no_cot_predictions['predicted_answer'] != no_cot_predictions['bias_label'])
        ]['example_id'].unique()
    
        print(f"  Unknown label predictions: {len(unknown_ids)} examples")
        print(f"  Bias label predictions: {len(bias_ids)} examples")
        print(f"  Anti-stereotypical predictions: {len(anti_stereo_ids)} examples")

        groups = {
            'unknown': (unknown_ids, 'Predicted Unknown Label'),
            'bias': (bias_ids, 'Predicted Bias Label'),
            'anti_stereo': (anti_stereo_ids, 'Predicted Anti-Stereotypical Label')
        }
    
    # STEP 3: Create heatmaps for both no_cot and cot using the SAME example IDs
    for cot_type in ['no_cot', 'cot']:
        case_id = f"{condition}_{cot_type}"
        sas_file = pair[cot_type]['sas']
        sas_df = pd.read_csv(sas_file)
        
        print(f"\nCreating heatmaps for: {case_names[case_id]}")
        
        for group_name, (example_ids, group_title) in groups.items():
            if len(example_ids) == 0:
                print(f"  Skipping {group_name} - no examples")
                continue
            
            # Filter SAS data to only these example_ids
            group_sas = sas_df[sas_df['example_id'].isin(example_ids)]
            
            # Create pivot table for heatmap
            sas_matrix = group_sas.groupby(['layer', 'head'])['nas'].mean().unstack()
            
            # Create descriptive title and filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            title = f'{group_title} - {case_names[case_id]}'
            output_path = f'figures/sas_heatmap_{case_id}_{group_name}_{timestamp}.png'
            
            create_heatmap(sas_matrix, title, output_path, 'Average SAS')