import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from sklearn.model_selection import train_test_split

# ✅ 1️⃣ Load & Prepare Dataset
input_path = "clean_sentences/cleaned_gujarati_cricket_sentences_2_to_15.csv"
df = pd.read_csv(input_path)

# Ensure column names are correct
df = df.rename(columns={"clean_sentence": "text"})

# Split into train & test (80%-20%)
train_texts, test_texts = train_test_split(df["text"].tolist(), test_size=0.2, random_state=42)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert", use_fast=False)

# Tokenization function with labels
def tokenize_function(examples):
    tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()  # MLM requires labels = input_ids
    return tokens

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict({"text": train_texts})
test_dataset = Dataset.from_dict({"text": test_texts})

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# ✅ 2️⃣ Load Pretrained indic-bert Model (MaskedLM)
model = AutoModelForMaskedLM.from_pretrained("ai4bharat/indic-bert")

# ✅ 3️⃣ Define Training Arguments
training_args = TrainingArguments(
    output_dir="./bert_gujarati_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
)

# ✅ 4️⃣ Define MLM Data Collator (Automatically masks tokens)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15  # 15% tokens are masked
)

# ✅ 5️⃣ Train the Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
)

trainer.train()

# ✅ 6️⃣ Save Fine-Tuned Model
model.save_pretrained("./bert_gujarati_finetuned")
tokenizer.save_pretrained("./bert_gujarati_finetuned")

print("\n✅ Fine-tuning complete! Model saved in './bert_gujarati_finetuned'")
