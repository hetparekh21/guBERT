import pandas as pd
import os
import torch
import argparse
import random
from datetime import datetime
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Custom dataset for MLM
class GujaratiTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten()
        }

def main():
    parser = argparse.ArgumentParser(description="Prepare Gujarati text for BERT MLM fine-tuning")
    parser.add_argument("--input", type=str, required=True, help="Path to cleaned CSV file with 'clean_sentence' column")
    parser.add_argument("--output_dir", type=str, help="Output directory (defaults to timestamped folder)")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length for tokenization")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of data to use for validation")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Probability of masking tokens for MLM")
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"gujarati_mlm_prep_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📊 Preparing Gujarati text for BERT MLM fine-tuning")
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Load the cleaned dataset
        print(f"📂 Loading data from: {args.input}")
        df = pd.read_csv(args.input)
        
        print(f"✅ Loaded {len(df)} rows from CSV")

        # Check for the clean_sentence column
        column_name = "clean_sentence"
        if column_name not in df.columns:
            # Try alternate column names
            if "cleaned_sentence" in df.columns:
                column_name = "cleaned_sentence"
            elif "sentence" in df.columns:
                column_name = "sentence"
                print("⚠️ Using 'sentence' column - assuming data is already cleaned")
            else:
                raise ValueError("CSV must contain a 'clean_sentence', 'cleaned_sentence', or 'sentence' column")
        
        # Get texts and remove any empty strings
        texts = df[column_name].dropna().tolist()
        texts = [text for text in texts if isinstance(text, str) and text.strip()]
        
        print(f"ℹ️ Total sentences: {len(texts)}")
        
        # Load IndicBERT tokenizer
        print("🔤 Loading IndicBERT tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
            print("✅ Loaded AI4Bharat's IndicBERT tokenizer")
        except Exception as e:
            print(f"⚠️ Error loading AI4Bharat tokenizer: {str(e)}")
            print("🔄 Falling back to Google's multilingual BERT tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            print("✅ Loaded multilingual BERT tokenizer")
        
        # Save tokenizer
        tokenizer_path = os.path.join(output_dir, "tokenizer")
        tokenizer.save_pretrained(tokenizer_path)
        print(f"💾 Saved tokenizer to: {tokenizer_path}")
        
        # Split into train/validation sets
        train_texts, val_texts = train_test_split(
            texts, test_size=args.test_size, random_state=42
        )
        
        print(f"ℹ️ Training examples: {len(train_texts)}")
        print(f"ℹ️ Validation examples: {len(val_texts)}")
        
        # Create datasets
        print(f"🧩 Creating MLM datasets with max_length={args.max_length}...")
        train_dataset = GujaratiTextDataset(train_texts, tokenizer, max_length=args.max_length)
        val_dataset = GujaratiTextDataset(val_texts, tokenizer, max_length=args.max_length)
        
        # Create data collator for MLM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=True, 
            mlm_probability=args.mlm_probability
        )
        
        # Save datasets
        train_path = os.path.join(output_dir, "train_dataset.pt")
        val_path = os.path.join(output_dir, "val_dataset.pt")
        
        torch.save(train_dataset, train_path)
        torch.save(val_dataset, val_path)
        print(f"💾 Saved train dataset to: {train_path}")
        print(f"💾 Saved validation dataset to: {val_path}")
        
        # Save a sample of tokenized sentences for inspection
        sample_size = min(5, len(train_texts))
        sample_file = os.path.join(output_dir, "tokenization_samples.txt")
        
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("TOKENIZATION SAMPLES\n")
            f.write("===================\n\n")
            
            for i in range(sample_size):
                text = train_texts[i]
                tokens = tokenizer.tokenize(text)
                token_ids = tokenizer.encode(text)
                
                f.write(f"SAMPLE {i+1}:\n")
                f.write(f"Text: {text}\n\n")
                f.write(f"Tokens ({len(tokens)}):\n{tokens}\n\n")
                f.write(f"Token IDs ({len(token_ids)}):\n{token_ids}\n\n")
                
                # Show a MLM example
                encoding = tokenizer(text, return_special_tokens_mask=True)
                input_ids = encoding["input_ids"]
                special_tokens_mask = encoding["special_tokens_mask"]
                
                # Create a simple MLM example manually
                mlm_input = input_ids.copy()
                probability_matrix = [0] * len(input_ids)
                
                for i in range(len(input_ids)):
                    if special_tokens_mask[i] == 1:
                        continue
                    probability_matrix[i] = random.random() < args.mlm_probability
                
                # Mask some of the tokens
                for i in range(len(input_ids)):
                    if probability_matrix[i] == 1:
                        mlm_input[i] = tokenizer.mask_token_id
                
                f.write("MLM Example:\n")
                f.write(f"Original: {tokenizer.decode(input_ids)}\n")
                f.write(f"Masked:   {tokenizer.decode(mlm_input)}\n\n")
                f.write("=" * 50 + "\n\n")
        
        print(f"📝 Saved tokenization samples to: {sample_file}")
                     
        print("\n✅ MLM Preparation Complete!")
        print(f"📁 All files saved to: {output_dir}")
        print(f"🚀 To train the model, run: python {train_script_path}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()