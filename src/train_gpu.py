import os, time, torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 128
BATCH_SIZE = 32
EPOCHS     = 3
LR         = 2e-5

os.makedirs("models/distilbert_abuse", exist_ok=True)


class AbuseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids"     : self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels"        : self.labels[idx],
        }


def train_distilbert(train_texts, train_labels, val_texts, val_labels):
    print(f"  Device     : {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM       : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    print(f"  Tokenizing {len(train_texts)} train + {len(val_texts)} val samples...")

    train_ds = AbuseDataset(train_texts, train_labels, tokenizer)
    val_ds   = AbuseDataset(val_texts,   val_labels,   tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)

    optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps  = len(train_loader) * EPOCHS
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    history       = []
    best_val_acc  = 0.0
    best_val_f1   = 0.0

    for epoch in range(EPOCHS):
        # ── Train ──────────────────────────────────────────────
        model.train()
        total_loss  = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()
            out  = model(
                input_ids      = batch["input_ids"].to(DEVICE),
                attention_mask = batch["attention_mask"].to(DEVICE),
                labels         = batch["labels"].to(DEVICE),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()

            if step % 100 == 0:
                print(f"    Epoch {epoch+1} | Step {step}/{len(train_loader)} | "
                      f"Loss: {total_loss/step:.4f}")

        avg_loss = total_loss / len(train_loader)

        # ── Validate ───────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                out    = model(
                    input_ids      = batch["input_ids"].to(DEVICE),
                    attention_mask = batch["attention_mask"].to(DEVICE),
                )
                preds  = torch.argmax(out.logits, dim=1).cpu().numpy()
                labels = batch["labels"].numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)

        from sklearn.metrics import accuracy_score, f1_score
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1  = f1_score(all_labels, all_preds, average="weighted")
        elapsed = time.time() - epoch_start

        print(f"\n  Epoch {epoch+1}/{EPOCHS}  Loss: {avg_loss:.4f}  "
              f"Val Acc: {val_acc:.4f}  Val F1: {val_f1:.4f}  ({elapsed:.1f}s)")

        history.append({"epoch": epoch+1, "loss": avg_loss,
                        "val_acc": val_acc, "val_f1": val_f1})

        if val_f1 > best_val_f1:
            best_val_acc = val_acc
            best_val_f1  = val_f1
            model.save_pretrained("models/distilbert_abuse")
            tokenizer.save_pretrained("models/distilbert_abuse")
            print(f"  => Best model saved  (F1={val_f1:.4f})\n")

    print(f"\n  Training complete.  Best Val F1: {best_val_f1:.4f}")
    return model, tokenizer, history
