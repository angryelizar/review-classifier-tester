import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import datasets.config as ds_config
from sklearn.metrics import classification_report, accuracy_score
ds_config.TORCHVISION_AVAILABLE = False
model_name = "ai-forever/ruBert-base"
dataset_name = "angryelizar/sentiment_dataset_splitted_short"
device = "cpu"

if (torch.backends.mps.is_available()):
    device = "mps"
    print("MPS device found. Using GPU acceleration.")

if (torch.cuda.is_available()):
    device = "cuda"
    print("CUDA device found. Using GPU acceleration.")


def main() -> None:
    print("Hello from review-classifier-tester!")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}\n")

    print("Loading dataset...")
    dataset = load_dataset(dataset_name)
    print(f"Dataset loaded. Test split size: {len(dataset['test'])} rows")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer loaded.")

    def tokenize(batch): return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=267)

    print("\nTaking 10% subsample of test split (stratified)...")
    short_dataset = dataset["test"].train_test_split(test_size=0.1, stratify_by_column="label", seed=42)
    print(f"Subsample size: {len(short_dataset['test'])} rows")

    print("Tokenizing...")
    tokenized = short_dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    print("Tokenization done.")

    test_loader = DataLoader(tokenized["test"], batch_size=32, shuffle=False, num_workers=2)
    total_batches = len(test_loader)
    print(f"Test loader ready: {total_batches} batches\n")

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    if (device != "cpu"): model = model.to(device)
    print("Model loaded and moved to device.\n")

    model.eval()
    print("Starting evaluation...")

    predictions = []
    true_labels = []

    for step, batch in enumerate(test_loader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            # Move logits and labels to CPU
            logits = logits.detach().cpu()
            label_ids = labels.to('cpu').numpy()

            preds = torch.argmax(logits, dim=-1).numpy()

            predictions.extend(preds)
            true_labels.extend(label_ids)

        if step % 10 == 0 or step == total_batches - 1:
            print(f"  Batch {step + 1}/{total_batches} — loss: {outputs.loss.item():.4f}")

    print("\nEvaluation loop finished.")
    print("\n=== Evaluation Metrics ===")
    print(f"Test Accuracy: {accuracy_score(true_labels, predictions):.4f}\n")
    print("Classification Report:")
    print(classification_report(true_labels, predictions, target_names=["Neutral", "Positive", "Negative"]))