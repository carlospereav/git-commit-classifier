import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

MODEL_PATH = "./model"

def predict():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have run 'scripts/02_train_model.py' first.")
        return

    # Test examples covering different classes
    test_messages = [
        "fix: prevent crash when user inputs empty string",
        "feat: add new dark mode toggle to settings",
        "docs: update installation instructions in README",
        "style: format code with black",
        "refactor: simplify authentication logic",
        "test: add unit tests for user login",
        "chore: update dependencies",
        "ci: fix github actions pipeline",
        "build: upgrade webpack version",
        "perf: optimize image loading speed",
        "this is a random commit message without prefix" # Tricky one
    ]

    print("\nRunning Inference Tests:\n")
    print(f"{'MESSAGE':<50} | {'PREDICTION':<10} | {'CONFIDENCE':<10}")
    print("-" * 80)

    model.eval()
    for msg in test_messages:
        inputs = tokenizer(msg, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            confidence, predicted_class_id = torch.max(probs, dim=-1)
            
        predicted_label = model.config.id2label[predicted_class_id.item()]
        confidence_score = confidence.item()

        print(f"{msg:<50} | {predicted_label:<10} | {confidence_score:.2%}")

if __name__ == "__main__":
    predict()
