
from math import ceil

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from tqdm.auto import tqdm

from digikala_training import make_loader
from digikala_transformer import BAND_LABELS, IGNORE_INDEX, REC_LABELS

def evaluate_csv(model, path, tokenizer, max_length, expected_rows, batch_size=256):
    targets = {
        "recommendation_status": REC_LABELS,
        "rating_band": BAND_LABELS,
    }
    actual = {target: [] for target in targets}
    predicted = {target: [] for target in targets}
    chunk_rows = 25_000
    total_rows = 0
    model.eval()

    chunks = pd.read_csv(
        path, chunksize=chunk_rows, dtype=str, keep_default_na=False
    )
    with torch.inference_mode():
        for chunk in tqdm(chunks, total=ceil(expected_rows / chunk_rows), desc="Evaluation"):
            total_rows += len(chunk)
            loader = make_loader(chunk, tokenizer, max_length, batch_size)
            for inputs, rec, band in loader:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(**inputs.to("cuda"))
                for target, scores, labels in zip(targets, logits, (rec, band)):
                    y = labels.numpy()
                    valid = y != IGNORE_INDEX
                    actual[target].append(y[valid])
                    predicted[target].append(scores.argmax(dim=1).cpu().numpy()[valid])

    assert total_rows == expected_rows, (total_rows, expected_rows)
    results = {"total_rows": total_rows, "heads": {}}
    for target, encoding in targets.items():
        y_true = np.concatenate(actual[target])
        y_pred = np.concatenate(predicted[target])
        names = sorted(encoding, key=encoding.get)
        codes = [encoding[name] for name in names]

        metrics = classification_report(
            y_true, y_pred, labels=codes, target_names=names,
            output_dict=True, zero_division=0,
        )
        matrix = confusion_matrix(y_true, y_pred, labels=codes)
        record = {
            "evaluated_rows": len(y_true),
            "missing_target_rows": total_rows - len(y_true),
            "metrics": metrics,
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "confusion_matrix": {"class_order": names, "counts": matrix.tolist()},
        }
        if target == "rating_band":
            record["mean_absolute_ordinal_distance"] = float(np.abs(y_true - y_pred).mean())
        results["heads"][target] = record
    return results
