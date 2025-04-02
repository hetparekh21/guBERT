import pandas as pd
from indicnlp.tokenize import sentence_tokenize

# Load dataset
file_path = "gujarati_cricket_data/21_to_200.csv"
df = pd.read_csv(file_path)

# Drop empty rows in the 'content' column
df = df.dropna(subset=["content"])

# Function to split paragraphs into sentences
def split_into_sentences(paragraph):
    return sentence_tokenize.sentence_split(paragraph, lang='gu')

# Apply function to dataset
df["sentences"] = df["content"].apply(split_into_sentences)

# Flatten the sentences into a list
sentences_list = [sentence for sublist in df["sentences"] for sentence in sublist]

# Convert to DataFrame
sentences_df = pd.DataFrame(sentences_list, columns=["sentence"])

# Save the processed dataset
output_path = "gujarati_cricket_sentences.csv"
sentences_df.to_csv(output_path, index=False)

# Display the first few sentences
print("Processed Dataset Sample:")
print(sentences_df.head())

print(f"\n✅ Processing complete! File saved as: {output_path}")
