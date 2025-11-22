import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Configuration
MODEL_NAME = "distilbert-base-uncased"
DATASET_NAME = "0x404/ccs_dataset"
OUTPUT_DIR = "./model"
LOG_DIR = "./results"

# Label Mapping (Alphabetical based on inspection)
LABEL2ID = {
    'build': 0,
    'chore': 1,
    'ci': 2,
    'docs': 3,
    'feat': 4,
    'fix': 5,
    'perf': 6,
    'refactor': 7,
    'style': 8,
    'test': 9
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    print(f"Starting training pipeline using {MODEL_NAME}...")

    # 2. Load Dataset
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)
    
    # 3. Preprocessing
    print("Preprocessing data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess_function(examples):
        # Tokenize text
        tokenized = tokenizer(examples["commit_message"], truncation=True, padding="max_length", max_length=64)
        # Map string labels to integers
        tokenized["label"] = [LABEL2ID[label] for label in examples["type"]]
        return tokenized

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # 4. Model Setup
    print("Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=LOG_DIR,
        num_train_epochs=3,              # Fast training for demo
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"], # Using 'test' split for validation during training
        compute_metrics=compute_metrics,
    )

    # 7. Train
    print("Training started...")
    trainer.train()

    # 8. Save Final Model
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
