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

    # Test examples WITHOUT prefixes (Hard Mode)
    # We want to see if the model understands the *content*, not just the prefix.
    test_messages = [
        "prevent crash when user inputs empty string",           # Should be: fix
        "add new dark mode toggle to settings",                  # Should be: feat
        "update installation instructions in README",            # Should be: docs
        "format code with black",                                # Should be: style
        "simplify authentication logic",                         # Should be: refactor
        "add unit tests for user login",                         # Should be: test
        "update dependencies",                                   # Should be: chore
        "fix github actions pipeline",                           # Should be: ci
        "upgrade webpack version",                               # Should be: build
        "optimize image loading speed",                          # Should be: perf
        "initial commit"                                         # Ambiguous
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
