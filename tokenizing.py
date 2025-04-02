import pandas as pd
from transformers import AutoTokenizer

# Load cleaned dataset
input_path = "clean_sentences/cleaned_gujarati_cricket_sentences_2_to_15.csv"
df = pd.read_csv(input_path,on_bad_lines='warn')

# Fix: Load IndicBERT tokenizer with SentencePiece
try:
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert", use_fast=False)
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")
    exit()

# Tokenization function
def tokenize_sentence(sentence):
    if isinstance(sentence, str):  # Ensure input is valid
        return tokenizer.encode(sentence, truncation=True, padding="max_length", max_length=128)
    return []  # Return empty list for invalid inputs

# Apply tokenization
df["tokenized"] = df["clean_sentence"].apply(tokenize_sentence)

# Remove empty tokenized rows (if any)
df = df[df["tokenized"].apply(len) > 0]

# Save the tokenized dataset
output_path = "tokenized_gujarati_cricket.json"
df[["clean_sentence", "tokenized"]].to_json(output_path, orient="records", lines=True)

# Show sample tokenized output
print("Tokenized Dataset Sample:")
print(df.head())

print(f"\n✅ Tokenization complete! File saved as: {output_path}")
