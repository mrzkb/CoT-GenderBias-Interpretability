import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

######## Modifications to separate prompts based on CoT change

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

def classify_prediction(row, is_disambig=False):
    """Classify a prediction as 'unknown', 'bias', or 'anti_stereo'"""
    pred = row['predicted_answer']
    unknown = row['unknown_label']
    bias = row['bias_label']
    
    if pred == unknown:
        return 'unknown'
    elif pred == bias:
        if is_disambig:
            # For disambig, also check correctness
            if pred == row['answer_label']:
                return 'bias_correct'
            else:
                return 'bias_incorrect'
        else:
            return 'bias'
    else:
        if is_disambig:
            # Anti-stereotypical with correctness
            if pred == row['answer_label']:
                return 'anti_correct'
            else:
                return 'anti_incorrect'
        else:
            return 'anti_stereo'

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
    
    # STEP 1: Load both no_cot and cot predictions
    no_cot_predictions = pd.read_csv(pair['no_cot']['predictions'])
    cot_predictions = pd.read_csv(pair['cot']['predictions'])
    
    # STEP 2: Classify each prediction
    no_cot_predictions['prediction_type'] = no_cot_predictions.apply(
        lambda row: classify_prediction(row, is_disambig), axis=1
    )
    cot_predictions['prediction_type'] = cot_predictions.apply(
        lambda row: classify_prediction(row, is_disambig), axis=1
    )
    
    # STEP 3: Merge to track changes
    # Merge on example_id to get both prediction types
    merged = no_cot_predictions[['example_id', 'prediction_type']].merge(
        cot_predictions[['example_id', 'prediction_type']], 
        on='example_id', 
        suffixes=('_no_cot', '_cot')
    )
    
    # Create transition label
    merged['transition'] = merged['prediction_type_no_cot'] + '_to_' + merged['prediction_type_cot']
    
    # STEP 4: Identify all unique transitions and their counts
    transition_counts = merged['transition'].value_counts()
    print(f"\nTransition counts:")
    for transition, count in transition_counts.items():
        print(f"  {transition}: {count} examples")
    
    # STEP 5: Create groups based on transitions
    transition_groups = {}
    for transition in merged['transition'].unique():
        example_ids = merged[merged['transition'] == transition]['example_id'].unique()
        # Create readable title
        parts = transition.split('_to_')
        group_title = f"{parts[0].replace('_', ' ').title()} → {parts[1].replace('_', ' ').title()}"
        transition_groups[transition] = (example_ids, group_title)
    
    # STEP 6: Create heatmaps for both no_cot and cot using transition-based groups
    for cot_type in ['no_cot', 'cot']:
        case_id = f"{condition}_{cot_type}"
        sas_file = pair[cot_type]['sas']
        sas_df = pd.read_csv(sas_file)
        
        print(f"\nCreating heatmaps for: {case_names[case_id]}")
        
        for transition, (example_ids, group_title) in transition_groups.items():
            if len(example_ids) == 0:
                print(f"  Skipping {transition} - no examples")
                continue
            
            # Filter SAS data to only these example_ids
            group_sas = sas_df[sas_df['example_id'].isin(example_ids)]
            
            if len(group_sas) == 0:
                print(f"  Warning: No SAS data found for {transition}")
                continue
            
            # Create pivot table for heatmap
            sas_matrix = group_sas.groupby(['layer', 'head'])['nas'].mean().unstack()
            
            # Create descriptive title and filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            title = f'{group_title} - {case_names[case_id]}'
            # Clean up transition name for filename
            transition_clean = transition.replace('_to_', '_TO_')
            output_path = f'figures/sas_heatmap_{case_id}_{transition_clean}_{timestamp}.png'
            
            create_heatmap(sas_matrix, title, output_path, 'Average SAS')

    # STEP 7: For each heatmap, identify top heads from no_cot and track their values
    print(f"\n{'='*60}")
    print(f"Creating CSV with top heads for {condition}")
    print(f"{'='*60}")
    
    for group_name, (example_ids, group_title) in transition_groups.items():
        if len(example_ids) == 0:
            continue
        
        # Get no_cot SAS for this group
        group_no_cot_sas = no_cot_sas[no_cot_sas['example_id'].isin(example_ids)]
        group_cot_sas = cot_sas[cot_sas['example_id'].isin(example_ids)]
        
        # Calculate average SAS per head for no_cot
        no_cot_avg = group_no_cot_sas.groupby(['layer', 'head'])['nas'].mean().reset_index()
        no_cot_avg.columns = ['layer', 'head', 'avg_nas']
        
        # Top 10 by absolute SAS
        no_cot_avg['abs_nas'] = no_cot_avg['avg_nas'].abs()
        top_10_abs = no_cot_avg.nlargest(10, 'abs_nas')[['layer', 'head', 'avg_nas']]
        
        # Top 5 most positive (stereotypical)
        top_5_pos = no_cot_avg.nlargest(5, 'avg_nas')[['layer', 'head', 'avg_nas']]
        
        # Top 5 most negative (anti-stereotypical)
        top_5_neg = no_cot_avg.nsmallest(5, 'avg_nas')[['layer', 'head', 'avg_nas']]
        
        # Create rows for CSV
        csv_rows = []
        
        # Add top 10 absolute
        for _, row in top_10_abs.iterrows():
            layer, head = int(row['layer']), int(row['head'])
            no_cot_val = row['avg_nas']
            
            # Get corresponding cot value
            cot_val = group_cot_sas[
                (group_cot_sas['layer'] == layer) & 
                (group_cot_sas['head'] == head)
            ]['nas'].mean()
            
            csv_rows.append({
                'case': 'top_10_absolute',
                'layer': layer,
                'head': head,
                'without_cot': no_cot_val,
                'cot': cot_val,
                'change': cot_val - no_cot_val
            })
        
        # Add top 5 positive (stereotypical)
        for _, row in top_5_pos.iterrows():
            layer, head = int(row['layer']), int(row['head'])
            no_cot_val = row['avg_nas']
            
            cot_val = group_cot_sas[
                (group_cot_sas['layer'] == layer) & 
                (group_cot_sas['head'] == head)
            ]['nas'].mean()
            
            csv_rows.append({
                'case': 'top_5_stereotypical',
                'layer': layer,
                'head': head,
                'without_cot': no_cot_val,
                'cot': cot_val,
                'change': cot_val - no_cot_val
            })
        
        # Add top 5 negative (anti-stereotypical)
        for _, row in top_5_neg.iterrows():
            layer, head = int(row['layer']), int(row['head'])
            no_cot_val = row['avg_nas']
            
            cot_val = group_cot_sas[
                (group_cot_sas['layer'] == layer) & 
                (group_cot_sas['head'] == head)
            ]['nas'].mean()
            
            csv_rows.append({
                'case': 'top_5_anti_stereotypical',
                'layer': layer,
                'head': head,
                'without_cot': no_cot_val,
                'cot': cot_val,
                'change': cot_val - no_cot_val
            })
        
        # Create DataFrame and save
        csv_df = pd.DataFrame(csv_rows)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        csv_path = f'figures/top_heads_{condition}_{group_name}_{timestamp}.csv'
        csv_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

# print("\n" + "="*60)
# print("All heatmaps generated successfully!")
# print("="*60)