
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from digikala_transformer import BAND_LABELS, IGNORE_INDEX, REC_LABELS, masked_cross_entropy
from normalize_persian import normalize_text

def make_loader(frame, tokenizer, max_length, batch_size, *, shuffle=False, seed=2026):
    bodies = frame["body"].map(normalize_text)
    recommendation = frame["recommendation_status"].map(REC_LABELS).fillna(IGNORE_INDEX)
    rating = frame["rating_band"].map(BAND_LABELS).fillna(IGNORE_INDEX)
    rows = list(zip(bodies, recommendation.astype(int), rating.astype(int)))

    def collate(batch):
        text, rec, band = zip(*batch)
        inputs = tokenizer(
            list(text), truncation=True, max_length=max_length,
            padding=True, pad_to_multiple_of=8, return_tensors="pt",
        )
        return inputs, torch.tensor(rec, dtype=torch.long), torch.tensor(band, dtype=torch.long)

    return DataLoader(
        rows, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )

def run_epoch(model, loader, weights, *, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = torch.zeros((), device="cuda")
    actual = [[], []]
    predicted = [[], []]

    for inputs, rec, band in tqdm(loader, desc="Train" if training else "Validation"):
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training), torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(**inputs.to("cuda"))
            loss = sum(
                masked_cross_entropy(scores, target.to("cuda"), weight)
                for scores, target, weight in zip(logits, (rec, band), weights)
            )

        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            for head, (scores, target) in enumerate(zip(logits, (rec, band))):
                valid = target.ne(IGNORE_INDEX)
                actual[head].append(target[valid].numpy())
                predicted[head].append(scores.argmax(dim=1).cpu()[valid].numpy())

        total_loss += loss.detach()

    result = {"loss": float(total_loss.item() / len(loader))}
    if not training:
        for head, name in enumerate(("recommendation_macro_f1", "rating_macro_f1")):
            result[name] = float(f1_score(
                np.concatenate(actual[head]), np.concatenate(predicted[head]),
                labels=[0, 1, 2], average="macro", zero_division=0,
            ))
    return result
