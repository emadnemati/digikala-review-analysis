
import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from digikala_transformer import DigikalaTransformer
from normalize_persian import normalize_text


class ReviewPredictor:
    def __init__(self, project_dir, device="cpu"):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use --device cpu.")

        models_dir = Path(project_dir) / "models"
        parsbert_dir = models_dir / "parsbert"
        self.config = json.loads(
            (models_dir / "inference_config.json").read_text(encoding="utf-8")
        )
        self.vectorizer = joblib.load(models_dir / "tfidf_vectorizer.joblib")
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.baselines = {
            target: joblib.load(models_dir / f"tfidf_{target}.joblib")
            for target in ["recommendation_status", "rating_band"]
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(parsbert_dir / "tokenizer"), local_files_only=True,
        )
        encoder = AutoModel.from_pretrained(
            str(parsbert_dir / "encoder"),
            local_files_only=True,
            use_safetensors=True,
            trust_remote_code=False,
            add_pooling_layer=False,
        )
        self.model = DigikalaTransformer(encoder)
        heads = load_file(str(parsbert_dir / "heads.safetensors"))
        for name in ["recommendation_head", "rating_head"]:
            getattr(self.model, name).load_state_dict({
                "weight": heads[f"{name}.weight"],
                "bias": heads[f"{name}.bias"],
            })
        self.model.to(self.device).eval()

    def _baseline_result(self, features, bundle):
        classifier = bundle["model"]
        probabilities = classifier.predict_proba(features)[0]
        order = np.argsort(-probabilities, kind="stable")
        winner, runner_up = int(order[0]), int(order[1])
        labels = [
            bundle["class_names"][int(code)]
            for code in classifier.classes_
        ]

        # Exact feature contributions to the winning-vs-runner-up logit margin.
        coefficient_margin = classifier.coef_[winner] - classifier.coef_[runner_up]
        contributions = coefficient_margin[features.indices] * features.data
        strongest = np.argsort(-np.abs(contributions))[:5]
        terms = [{
            "term": str(self.feature_names[features.indices[index]]),
            "logit_margin_contribution": float(contributions[index]),
        } for index in strongest]

        return {
            "label": labels[winner],
            "probabilities": {
                label: float(probability)
                for label, probability in zip(labels, probabilities)
            },
            "explanation": {
                "runner_up": labels[runner_up],
                "bias_margin": float(
                    classifier.intercept_[winner] - classifier.intercept_[runner_up]
                ),
                "strongest_terms": terms,
                "positive_contribution_favors": labels[winner],
            },
        }

    def predict(self, text):
        started = perf_counter()
        normalized = normalize_text(text)
        if not normalized:
            raise ValueError("Text is empty after normalization.")

        features = self.vectorizer.transform([normalized])
        result = {
            "model_version": self.config["model_version"],
            "text": text,
            "normalized_text": normalized,
            "tfidf": {
                target: self._baseline_result(features, bundle)
                for target, bundle in self.baselines.items()
            },
            "parsbert": {},
        }

        original_ids = self.tokenizer(
            normalized, truncation=False, verbose=False,
        )["input_ids"]
        inputs = self.tokenizer(
            normalized,
            truncation=True,
            max_length=self.config["max_length"],
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        for target, logits, encoding in zip(
            ["recommendation_status", "rating_band"],
            outputs,
            [
                self.config["recommendation_labels"],
                self.config["rating_band_labels"],
            ],
        ):
            probabilities = logits.float().softmax(dim=1)[0].cpu().numpy()
            names = sorted(encoding, key=encoding.get)
            result["parsbert"][target] = {
                "label": names[int(probabilities.argmax())],
                "probabilities": {
                    name: float(probabilities[encoding[name]])
                    for name in names
                },
            }

        result["token_count_before_truncation"] = len(original_ids)
        result["token_count_used"] = int(inputs["attention_mask"].sum().item())
        result["truncated"] = len(original_ids) > self.config["max_length"]
        result["latency_ms"] = round((perf_counter() - started) * 1000, 2)
        return result


def main():
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--text")
    choice.add_argument("--demo", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    predictor = ReviewPredictor(Path(__file__).resolve().parent, args.device)
    texts = [
        "خوبه",
        "معمولی بود، هنوز مطمئن نیستم ارزش خرید داشته باشه.",
        "بعد از دو روز خراب شد و اصلاً پیشنهاد نمی‌کنم.",
        "کیفیت ساخت خوبه ولی باتری زود تموم میشه و قیمتش بالاست.",
    ] if args.demo else [args.text]

    try:
        results = [predictor.predict(text) for text in texts]
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(results if args.demo else results[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
