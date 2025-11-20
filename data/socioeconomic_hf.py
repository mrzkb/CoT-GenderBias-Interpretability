import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import login

login()
# ---------------------------------------
# 1. Load your local dataframe
# ---------------------------------------
df = pd.read_csv("/scratch/osso6500/datasets/git/CoT-GenderBias-Interpretability/data/Socioeconomic_QA.csv")   # <-- replace with the file root

print("Initial size:", df.shape)

# ---------------------------------------
# 2. Shuffle the dataframe
# ---------------------------------------
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ---------------------------------------
# 3. Split: 200 test, 200 validation, rest train
# ---------------------------------------
test_df = df.iloc[:200].reset_index(drop=True)
valid_df = df.iloc[200:400].reset_index(drop=True)
train_df = df.iloc[400:].reset_index(drop=True)

print("Train:", train_df.shape)
print("Valid:", valid_df.shape)
print("Test :", test_df.shape)

# ---------------------------------------
# 4. Convert to HuggingFace DatasetDict
# ---------------------------------------
train_ds = Dataset.from_pandas(train_df, preserve_index=False)
valid_ds = Dataset.from_pandas(valid_df, preserve_index=False)
test_ds  = Dataset.from_pandas(test_df,  preserve_index=False)

dataset_dict = DatasetDict({
    "train": train_ds,
    "validation": valid_ds,
    "test": test_ds,
})

print(dataset_dict)

# ---------------------------------------
# 5. Login to Hugging Face
# ---------------------------------------
# If not logged in already:
# login()  # optional

# ---------------------------------------
# 6. Push to your repo
# ---------------------------------------
dataset_dict.push_to_hub("sophiaos/SocioeconomicQA") #replcae ??? with your env name
print("Dataset successfully pushed to Hugging Face!")