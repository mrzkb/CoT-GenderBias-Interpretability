from datasets import load_dataset
import pandas as pd

##### BEGIN
print(f"\n{'='*50}")
print("DATA LOADING")
print(f"{'='*50}")

# get 10% of the bbq dataset (randomly sample 5% ambig, 5% disambig)
bbq_gender_ds = load_dataset("Elfsong/BBQ", split="gender_identity")
bbq_gender_df = bbq_gender_ds.to_pandas()
print(f"Total BBQ gender dataset size: {len(bbq_gender_df)}")

# get 5% of the bbq dataset (randomly sample 2.5% ambig, 2.5% disambig)
ambig_sample = bbq_gender_df[bbq_gender_df['context_condition'] == 'ambig'].sample(n=150, random_state=22)
disambig_sample = bbq_gender_df[bbq_gender_df['context_condition'] == 'disambig'].sample(n=150, random_state=22)
print(f"Ambiguous samples: {len(ambig_sample)}")
print(f"Disambiguous samples: {len(disambig_sample)}")

# Save just in case!
ambig_sample[['example_id', 'context_condition', 'context', 'question', 'ans0', 'ans1', 'ans2', 'answer_label']].to_csv('data/prompts_sensitivity_ambig.csv', index=False)
disambig_sample[['example_id', 'context_condition', 'context', 'question', 'ans0', 'ans1', 'ans2', 'answer_label']].to_csv('data/prompts_sensitivity_disambig.csv', index=False)
print(f"Files saved to data/ directory")

# | 'example_id' | 'context_condition' | 'context' | 'question' | 'ans0' | 'ans1' | 'ans2' | 'answer_label' |