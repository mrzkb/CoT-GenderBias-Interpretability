import pandas as pd
import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def calc_sensitivity_metrics(df):
    num_questions = len(df)
    
    return {
        'forwards_accuracy': round((df['forwards_both_answer'] == df['answer_label']).sum() / num_questions, 2),
        'backwards_accuracy': round((df['backwards_both_answer'] == df['answer_label']).sum() / num_questions, 2),
        'position_sensitivity': round((df['forwards_both_answer'] != df['backwards_both_answer']).sum() / num_questions, 2),
        'backwards_tokens_accuracy': round((df['backwards_tokens_answer'] == df['answer_label']).sum() / num_questions, 2),
        'token_sensitivity': round((df['forwards_both_answer'] != df['backwards_tokens_answer']).sum() / num_questions, 2),
        'backwards_content_accuracy': round((df['backwards_content_answer'] == df['answer_label']).sum() / num_questions, 2),
        'selection_sensitivity': round((df['forwards_both_answer'] != df['backwards_content_answer']).sum() / num_questions, 2)
    }

def calc_priors_vectorized(df):
    """Calculate priors using vectorized operations - much faster."""
    # Stack the log probabilities into arrays
    log_p_prior0 = np.column_stack([
        df['zeroth_0_prob'], 
        df['first_2_prob'], 
        df['second_1_prob']
    ]).mean(axis=1)
    
    log_p_prior1 = np.column_stack([
        df['zeroth_1_prob'], 
        df['first_0_prob'], 
        df['second_2_prob']
    ]).mean(axis=1)
    
    log_p_prior2 = np.column_stack([
        df['zeroth_2_prob'], 
        df['first_1_prob'], 
        df['second_0_prob']
    ]).mean(axis=1)
    
    # Stack all priors and apply softmax across options (axis=1)
    log_priors = np.column_stack([log_p_prior0, log_p_prior1, log_p_prior2])
    priors = softmax(log_priors, axis=1)
    
    # Return average priors across all questions
    return {
        'prior(0)': round(priors[:, 0].mean(), 2),
        'prior(1)': round(priors[:, 1].mean(), 2),
        'prior(2)': round(priors[:, 2].mean(), 2)
    }

def evaluate_model_sensitivity(df):
    """Compute all sensitivity metrics without modifying input dataframe."""
    print(f"\nCalculating sensitivity metrics...")
    
    # Calculate metrics
    sensitivity_metrics = calc_sensitivity_metrics(df)
    prior_metrics = calc_priors_vectorized(df)
    
    # Combine into a single-row dataframe
    metrics = {**sensitivity_metrics, **prior_metrics}
    return pd.DataFrame([metrics])

def process_case(filepath, case_name):
    """Process a single case file."""
    df = pd.read_csv(filepath)
    sensitivity = evaluate_model_sensitivity(df)
    sensitivity['case'] = case_name
    return sensitivity

# Process all cases
cases = [
    ('data/responses_sensitivity_ambig_no_cot.csv', 'Ambiguous_NO_COT'),
    ('data/responses_sensitivity_ambig_cot.csv', 'Ambiguous_COT'),
    ('data/responses_sensitivity_disambig_no_cot.csv', 'Disambiguous_NO_COT'),
    ('data/responses_sensitivity_disambig_cot.csv', 'Disambiguous_COT')
]

all_cases_df = pd.concat([process_case(path, name) for path, name in cases], ignore_index=True)
all_cases_df.to_csv('outputs/sensitivity_llama.csv', index=False)

# def evaluate_model_sensitivity(df):
#     print(f"\nCalculating sensitivity metrics...")
#     df, priors = calc_priors(df)
    
#     sensitivity = pd.DataFrame()
#     sensitivity['position_sensitivity'] = calc_PS(df)
#     sensitivity['token_sensitivity'] = calc_TS(df)
#     sensitivity['selection_sensitivity'] = calc_SS(df)
#     sensitivity['prior(0)'] = priors[0]
#     sensitivity['prior(1)'] = priors[1]
#     sensitivity['prior(2)'] = priors[2]
   
#     return df, sensitivity

# def calc_PS(df):
#     # position sensitivity measured by the fluctuation rate
#     # The numbers of instances where the model’s response 
#     # to the forward version differs from the both_backwards version
#     # divided by the number of questions
#     num = (df['forwards_both_answer'] != df['backwards_both_answer']).sum()
#     den = len(df)
#     return num / den

# def calc_TS(df):
#     # token sensitivity measured by the fluctuation rate
#     # The numbers of instances where the model’s response 
#     # to the forward version differs from the backwards_token version
#     # divided by the number of questions
#     num = (df['forwards_both_answer'] != df['backwards_tokens_answer']).sum()
#     den = len(df)
#     return num / den

# def calc_SS(df):
#     # selection sensitivity (postiion and token) measured by the fluctuation rate
#     # The numbers of instances where the model’s response 
#     # to the forward version differs from the backwards_content version
#     # divided by the number of questions
#     num = (df['forwards_both_answer'] != df['backwards_content_answer']).sum()
#     den = len(df)
#     return num / den

# def calc_priors(df):
#     # zeroth: 0/1/2
#     # first rotation: 1/2/0
#     # second rotation: 2/0/1

#     # apply softmax
#     def softmax(x):
#         e_x = np.exp(x - np.max(x))
#         return e_x / e_x.sum(axis=0)

#     p_prior0_list = []
#     p_prior1_list = []
#     p_prior2_list = []

#     for idx, row in df.iterrows():
#         # Average the log probabilities for a given option for a single question
#         log_p_prior0 = (row['zeroth_0_prob'] + row['first_2_prob'] + row['second_1_prob']) / 3
#         log_p_prior1 = (row['zeroth_1_prob'] + row['first_0_prob'] + row['second_2_prob']) / 3
#         log_p_prior2 = (row['zeroth_2_prob'] + row['first_1_prob'] + row['second_0_prob']) / 3

#         # Apply softmax to the three log priors
#         p_prior0, p_prior1, p_prior2 = softmax(np.array([log_p_prior0, log_p_prior1, log_p_prior2]))

#         # Store in lists
#         p_prior0_list.append(p_prior0)
#         p_prior1_list.append(p_prior1)
#         p_prior2_list.append(p_prior2)
    
#     # Add the p_priors as new columns to the dataframe
#     df['p_prior0'] = p_prior0_list
#     df['p_prior1'] = p_prior1_list
#     df['p_prior2'] = p_prior2_list

#     # Calculate and return the averages
#     avg_p_prior0 = df['p_prior0'].mean()
#     avg_p_prior1 = df['p_prior1'].mean()
#     avg_p_prior2 = df['p_prior2'].mean()

#     return df, [avg_p_prior0, avg_p_prior1, avg_p_prior2]

# A_NCOT = pd.read_csv('data/responses_sensitivity_ambig_no_cot.csv')
# A_NCOT, A_NCOT_sensitivity = evaluate_model_sensitivity(A_NCOT)

# A_COT = pd.read_csv('data/responses_sensitivity_ambig_cot.csv')
# A_COT, A_COT_sensitivity = evaluate_model_sensitivity(A_COT)

# D_NCOT = pd.read_csv('data/responses_sensitivity_disambig_no_cot.csv')
# D_NCOT, D_NCOT_sensitivity = evaluate_model_sensitivity(D_NCOT)

# D_COT = pd.read_csv('data/responses_sensitivity_disambig_cot.csv')
# D_COT, D_COT_sensitivity = evaluate_model_sensitivity(D_COT)

# # all_cases = [A_NCOT_sensitivity, A_COT_sensitivity, D_NCOT_sensitivity, D_COT_sensitivity]
# # all_cases_df = pd.DataFrame(all_cases)
# # all_cases_df['case'] = ['Ambiguous_NO_COT', 'Ambiguous_COT', 'Disambiguous_NO_COT', 'Disambiguous_COT']

# all_cases_df = pd.concat([
#     A_NCOT_sensitivity.assign(case='Ambiguous_NO_COT'),
#     A_COT_sensitivity.assign(case='Ambiguous_COT'),
#     D_NCOT_sensitivity.assign(case='Disambiguous_NO_COT'),
#     D_COT_sensitivity.assign(case='Disambiguous_COT')
# ], ignore_index=True)

# all_cases_df.to_csv(f'outputs/sensitivity_llama.csv', index=False)