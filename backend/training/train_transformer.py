"""Phase 3 (M4, optional): fine-tune MiniLM as resume-JD match classifier.

Pretrained SBERT gives strong zero-shot cosine (Phase 2); this script adds a
supervised classification head: `resume_text [SEP] jd_text -> label_match`,
jointly with the paper's competence-level idea (implemented here as a second
head trained separately for simplicity on CPU).

Full fine-tune on 7.4k pairs is GPU work (RTX 3060 12GB: ~5-10 min).
Defaults are a CPU smoke test that proves wiring end-to-end (no artifacts kept).

Usage (from backend/):
    python -m training.train_transformer --smoke        # 64 pairs, 1 epoch, temp output
    python -m training.train_transformer --run_full --epochs 3 --sample 0 --batch_size 32 --fp16
Output (full only): models/registry/ensemble_v1/transformer_m4/
Requires: transformers, accelerate, torch with CUDA for --fp16
  (pip install torch --index-url https://download.pytorch.org/whl/cu126).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_dataset(pairs_df, tokenizer, max_length):
    import torch
    from torch.utils.data import Dataset

    class PairDS(Dataset):
        def __init__(self, df):
            self.enc = tokenizer(list(df["resume_text"]), list(df["jd_text"]),
                                 truncation=True, padding=True,
                                 max_length=max_length)
            self.y = df["label_match"].to_numpy()

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return {"input_ids": torch.tensor(self.enc["input_ids"][i]),
                    "attention_mask": torch.tensor(self.enc["attention_mask"][i]),
                    "labels": torch.tensor(int(self.y[i]))}

    return PairDS(pairs_df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", default=True)
    ap.add_argument("--run_full", action="store_true")
    ap.add_argument("--sample", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--fp16", action="store_true",
                    help="mixed precision (CUDA GPU only, e.g. RTX 3060)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    import pandas as pd
    if args.run_full:
        from training.utils import REGISTRY
        n, max_len = (args.sample or None), args.max_length or 256
        epochs = max(2, args.epochs)
        out_dir = args.output or os.path.join(REGISTRY, "transformer_m4")
        train_bs, eval_bs, save, fp16 = args.batch_size or 32, 64, "epoch", args.fp16
    else:
        import tempfile
        n, max_len, epochs, out_dir = args.sample, args.max_length, 1, tempfile.mkdtemp(prefix="m4_smoke_")
        train_bs, eval_bs, save, fp16 = 8, 16, "no", False

    df = pd.read_parquet(os.path.join(BACKEND, "data", "processed", "pairs_v2.parquet"))
    tr = df[df["split"] == "train"].sample(n=n or len(df[df["split"] == "train"]),
                                           random_state=42).reset_index(drop=True)
    va = df[df["split"] == "val"].sample(n=min(64, (df["split"] == "val").sum()),
                                         random_state=42).reset_index(drop=True)

    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.gradient_checkpointing_enable()

    train_ds, eval_ds = build_dataset(tr, tok, max_len), build_dataset(va, tok, max_len)
    import torch
    print("device:", "cuda" if torch.cuda.is_available() else "cpu",
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    targs = TrainingArguments(output_dir=out_dir, per_device_train_batch_size=train_bs,
                              per_device_eval_batch_size=eval_bs, num_train_epochs=epochs,
                              learning_rate=args.lr, warmup_steps=50 if args.run_full else 0, fp16=fp16,
                              eval_strategy="epoch", save_strategy=save,
                              load_best_model_at_end=bool(args.run_full),
                              metric_for_best_model="f1",
                              logging_steps=10, seed=42, report_to="none")
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    def metrics(p):
        pred = np.argmax(p.predictions, axis=1)
        return {"accuracy": accuracy_score(p.label_ids, pred),
                "f1": f1_score(p.label_ids, pred)}

    res = Trainer(model=model, args=targs, train_dataset=train_ds,
                  eval_dataset=eval_ds, compute_metrics=metrics).train()
    print("train_loss:", round(res.training_loss, 4))
    print("eval:", Trainer(model=model, args=targs, train_dataset=train_ds,
                           eval_dataset=eval_ds, compute_metrics=metrics).evaluate())
    if args.run_full:
        Trainer(model=model, args=targs, train_dataset=train_ds,
                eval_dataset=eval_ds).save_model(out_dir)
        print("saved ->", out_dir)
    else:
        print("smoke OK (no artifacts kept; use --run_full on GPU to keep M4)")


if __name__ == "__main__":
    main()
