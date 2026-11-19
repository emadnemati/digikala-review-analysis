#!/usr/bin/env python3
"""Finalize the existing Digikala project in Google Colab."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_ROOT = Path("/content/drive/MyDrive/digikala-project-3")
NOTEBOOKS = [
    "01_descriptive_analysis.ipynb",
    "02_baselines.ipynb",
    "03_transfer_learning_finetuning.ipynb",
    "04_semi_supervised.ipynb",
]
TEST_REPORTS = {
    "TF-IDF": "baseline_test.json",
    "ParsBERT 1.2M": "parsbert_supervised_1200k_test.json",
    "Semi-supervised 1.2M": "parsbert_semi_supervised_1200k_test.json",
    "ParsBERT 3M": "parsbert_supervised_3000k_test.json",
}
SPLIT_FILES = [
    "train_supervised.csv",
    "train_pseudo_unlabeled.csv",
    "natural_unlabeled.csv",
    "validation.csv",
    "test.csv",
]
EXPERIMENTS = [
    "parsbert_supervised",
    "parsbert_supervised_1200k",
    "parsbert_supervised_3000k",
    "parsbert_semi_supervised_1200k",
]
MODULES = [
    "digikala_transformer.py",
    "digikala_training.py",
    "digikala_evaluation.py",
    "normalize_persian.py",
]
BASELINE_FILES = [
    "tfidf_vectorizer.joblib",
    "tfidf_recommendation_status.joblib",
    "tfidf_rating_band.joblib",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def collect_notebooks(root, release):
    folders = [
        root / "notebooks",
        Path("/content"),
        root,
        Path("/content/drive/MyDrive/Colab Notebooks"),
        Path("/content/drive/MyDrive"),
    ]
    inventory = {}
    target_dir = release / "notebooks"
    target_dir.mkdir(exist_ok=True)
    for name in NOTEBOOKS:
        matches = []
        stem = name.removesuffix(".ipynb").casefold()
        for folder in folders:
            if not folder.is_dir():
                continue
            for path in folder.iterdir():
                if path.name.removesuffix(".ipynb").casefold() == stem and path.is_file():
                    notebook = read_json(path)
                    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
                        raise ValueError(f"Invalid notebook: {path}")
                    matches.append(path)
        if matches:
            source = max(matches, key=lambda path: path.stat().st_mtime)
            notebook = read_json(source)
            shutil.copyfile(source, target_dir / name)
            inventory[name] = {
                "source": str(source),
                "cells": len(notebook["cells"]),
                "sha256": sha256_file(target_dir / name),
            }
        elif (target_dir / name).is_file():
            notebook = read_json(target_dir / name)
            if notebook.get("nbformat") == 4 and isinstance(notebook.get("cells"), list):
                inventory[name] = {
                    "source": str(target_dir / name),
                    "cells": len(notebook["cells"]),
                    "sha256": sha256_file(target_dir / name),
                }
    return inventory


def inspect_data(splits, reports):
    manifest = read_json(splits / "split_manifest.json")

    # The original dataset should already be in the cache from error analysis.
    from huggingface_hub import hf_hub_download

    raw_path = Path(hf_hub_download(
        repo_id="RadeAI/Digikala_comments_products",
        filename="digikala-comments.csv",
        repo_type="dataset",
    ))
    print("Recording the full-file schema and source checksum...", flush=True)
    with duckdb.connect(config={"memory_limit": "3GB", "threads": "2"}) as db:
        db.execute(
            "SELECT * FROM read_csv(?, header=true, sample_size=-1) LIMIT 0",
            [str(raw_path)],
        )
        schema = pd.DataFrame([
            {"column": field[0], "inferred_type": str(field[1])}
            for field in db.description
        ])
    if len(schema) != 15:
        raise ValueError(f"Unexpected source schema: {len(schema)} columns")
    schema.to_csv(reports / "dataset_schema.csv", index=False)
    source_record = {
        "repository": "RadeAI/Digikala_comments_products",
        "filename": raw_path.name,
        "bytes": raw_path.stat().st_size,
        "sha256": sha256_file(raw_path),
    }

    print("Measuring exact-text overlap for the selected 3M training subset...", flush=True)
    with duckdb.connect(config={"memory_limit": "3GB", "threads": "2"}) as db:
        row = db.execute("""
            WITH selected AS MATERIALIZED (
                SELECT id, body
                FROM read_csv(?, header=true, all_varchar=true)
                ORDER BY substr(sha256('2026|' || id), 1, 16), id
                LIMIT 3000000
            ), train_texts AS MATERIALIZED (
                SELECT DISTINCT body FROM selected
            )
            SELECT count(*) AS test_rows,
                   count(t.body) AS shared_text_rows
            FROM read_csv(?, header=true, all_varchar=true) e
            LEFT JOIN train_texts t USING (body)
        """, [
            str(splits / "train_supervised.csv"), str(splits / "test.csv")
        ]).fetchone()
    if row[0] != 919_675:
        raise ValueError(f"Unexpected test count: {row[0]}")

    overlap = {
        "training_rows": 3_000_000,
        "test_rows": int(row[0]),
        "test_rows_with_text_in_selected_train": int(row[1]),
        "test_overlap_pct": 100 * row[1] / row[0],
    }
    write_json(reports / "selected_3m_text_overlap.json", overlap)
    manifest["source_file"] = source_record
    manifest["selected_3m_text_overlap"] = overlap
    manifest["cleaning_policy"] = {
        "missing_text_markers": ["", "nan"],
        "multiple_texts_for_same_id": "Exclude the complete ID group",
        "same_text_for_same_id": "Merge to one row",
        "conflicting_recommendations": "Mask recommendation target",
        "conflicting_rating_bands": "Mask rating-band target",
        "invalid_rate": "Mask rating-band target",
        "training_rows_without_either_target": "Exclude from train_supervised",
    }
    return manifest, overlap, schema


def draw_comparison(results, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, target, title in zip(
        axes,
        ["recommendation_status", "rating_band"],
        ["Recommendation — test macro F1", "Rating band — test macro F1"],
    ):
        values = [
            results[name]["heads"][target]["metrics"]["macro avg"]["f1-score"]
            for name in names
        ]
        bars = ax.barh(
            names, values, color=["#687787", "#687787", "#687787", "#235789"]
        )
        ax.invert_yaxis()
        ax.set(title=title, xlim=(0, 1), xlabel="Macro F1")
        ax.bar_label(bars, fmt="%.4f", padding=4)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def load_report_metadata(release):
    """Read measured results without regenerating the reviewed documents."""
    reports = release / "reports"
    documents = [
        release / "README.md", reports / "final_report.md",
        reports / "final_report.pdf", reports / "error_analysis.md",
    ]
    missing = [str(path) for path in documents if not path.is_file()]
    if missing:
        raise FileNotFoundError("Reviewed documents missing: " + ", ".join(missing))
    return (
        read_json(reports / "baseline_training.json"),
        read_json(reports / "semi_supervised_test_comparison.json"),
        read_json(reports / "pseudo_label_accuracy_1200k.json"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "reports").is_dir():
        raise FileNotFoundError(f"Project reports folder not found: {root / 'reports'}")
    release = root / "release"
    reports = root / "reports"
    release.mkdir(exist_ok=True)

    # These files were produced and tested by the preceding offline-export cells.
    existing_assets = [
        "predict.py", "requirements.txt", "models/inference_config.json",
        "models/parsbert/encoder/config.json",
        "models/parsbert/encoder/model.safetensors",
        "models/parsbert/heads.safetensors",
        *[f"models/{name}" for name in BASELINE_FILES],
    ]
    absent_assets = [name for name in existing_assets if not (release / name).is_file()]
    if absent_assets:
        raise FileNotFoundError("Offline export files missing: " + ", ".join(absent_assets))

    results = {name: read_json(reports / filename) for name, filename in TEST_REPORTS.items()}
    if {item["total_rows"] for item in results.values()} != {919_675}:
        raise ValueError("Model reports do not refer to the same full test set.")
    for target in ["recommendation_status", "rating_band"]:
        counts = {item["heads"][target]["evaluated_rows"] for item in results.values()}
        if len(counts) != 1:
            raise ValueError(f"Different evaluated populations for {target}")

    configuration = read_json(release / "models" / "inference_config.json")
    if configuration["checkpoint_sha256"] != results["ParsBERT 3M"]["checkpoint_sha256"]:
        raise ValueError("Exported model metadata differs from the tested 3M checkpoint.")
    selected_checkpoint = root / "artifacts" / "parsbert_supervised_3000k" / "best.pt"
    if sha256_file(selected_checkpoint) != configuration["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint checksum differs from the test report.")

    errors = pd.concat([
        pd.read_csv(reports / name, dtype={"id": str}, keep_default_na=False)
        for name in ["error_cases_for_review.csv", "sarcasm_candidates.csv"]
    ], ignore_index=True)
    if len(errors) != 32 or not errors["id"].is_unique:
        raise ValueError("Expected 32 distinct reviewed error cases.")
    if not errors["manually_reviewed"].astype(str).str.lower().eq("true").all():
        raise ValueError("Some error cases have not been reviewed.")
    demo = read_json(reports / "offline_demo.json")
    if len(demo) != 4:
        raise ValueError("Expected four offline demonstration results.")

    splits = Path("/content/digikala_splits")
    if not all((splits / name).is_file() for name in [*SPLIT_FILES, "split_manifest.json"]):
        print("Restoring split files...", flush=True)
        bundle = Path("/content/digikala_inputs.tar.gz")
        shutil.copyfile(root / bundle.name, bundle)
        with tarfile.open(bundle, "r:gz") as archive:
            archive.extractall("/content", filter="data")

    manifest, overlap, schema = inspect_data(splits, reports)
    data_dir = release / "data"
    data_dir.mkdir(exist_ok=True)
    for name in SPLIT_FILES:
        print(f"Copying {name}...", flush=True)
        shutil.copyfile(splits / name, data_dir / name)
    write_json(data_dir / "split_manifest.json", manifest)

    print("Copying experiment artifacts...", flush=True)
    for name in EXPERIMENTS:
        shutil.copytree(
            root / "artifacts" / name, release / "artifacts" / name,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("*.tmp", "__pycache__"),
        )
    for name in BASELINE_FILES:
        shutil.copyfile(root / "artifacts" / name, release / "artifacts" / name)
    for name in MODULES:
        shutil.copyfile(root / "artifacts" / "parsbert_supervised" / name, release / name)
    finalizer_source = Path(__file__).resolve()
    finalizer_destination = release / "finalize_digikala_project.py"
    if finalizer_source != finalizer_destination.resolve():
        shutil.copyfile(finalizer_source, finalizer_destination)
    for name in ["digikala_inputs.tar.gz", "digikala_project.duckdb"]:
        shutil.copyfile(root / name, release / name)
    shutil.copytree(reports, release / "reports", dirs_exist_ok=True)
    shutil.copyfile(root / "README.md", release / "README.md")

    inventory = collect_notebooks(root, release)
    figures = release / "reports" / "figures"
    draw_comparison(results, figures / "test_macro_f1.png")
    baseline_training, comparison, audit = load_report_metadata(release)

    print("Checking the standalone offline --text command on CPU...", flush=True)
    process = subprocess.run(
        [
            sys.executable, str(release / "predict.py"), "--text",
            "کیفیتش خوبه ولی قیمتش بالاست.", "--device", "cpu",
        ],
        cwd=release,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        capture_output=True, text=True, encoding="utf-8", timeout=300,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr)
    prediction = json.loads(process.stdout)
    if set(prediction["parsbert"]) != {"recommendation_status", "rating_band"}:
        raise ValueError("Unexpected prediction output.")
    for model_name in ["tfidf", "parsbert"]:
        for target in ["recommendation_status", "rating_band"]:
            probabilities = prediction[model_name][target]["probabilities"]
            if abs(sum(probabilities.values()) - 1) > 1e-5:
                raise ValueError("Class probabilities do not sum to one.")
            if not all(0 <= value <= 1 for value in probabilities.values()):
                raise ValueError("Invalid probability.")

    required = [
        *existing_assets, "README.md", "finalize_digikala_project.py", *MODULES,
        "reports/final_report.md", "reports/final_report.pdf", "reports/error_analysis.md",
        "reports/offline_demo.json", "data/split_manifest.json",
        *[f"notebooks/{name}" for name in NOTEBOOKS],
        *[f"data/{name}" for name in SPLIT_FILES],
    ]
    missing = [name for name in required if not (release / name).is_file()]
    release_report = {
        "created_at": datetime.now(ZoneInfo("Asia/Tehran")).isoformat(timespec="seconds"),
        "status": "ready" if not missing else "incomplete",
        "selected_model": "ParsBERT 3M",
        "selected_test_results": results["ParsBERT 3M"],
        "semi_supervised_comparison": comparison,
        "pseudo_label_audit": audit,
        "reviewed_error_cases": len(errors),
        "notebooks": inventory,
        "missing_files": missing,
        "offline_cpu_demo_passed": True,
        "offline_cpu_text_command_passed": True,
        "full_notebook_rerun_verified": False,
        "exact_text_overlap_selected_3m": overlap,
        "baseline_training": baseline_training,
    }
    write_json(release / "release_report.json", release_report)
    if missing:
        print("\nREPORTS_READY_BUT_FILES_MISSING", flush=True)
        for name in missing:
            print(f"MISSING: {name}", flush=True)
        print("Upload the missing notebooks to /content and run this script again.", flush=True)
        return 2

    print("Generating file checksums...", flush=True)
    payload = sorted(
        path for path in release.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
        and not path.name.endswith(".tmp") and "__pycache__" not in path.parts
    )
    checksum_path = release / "checksums.sha256"
    with checksum_path.open("w", encoding="utf-8") as file:
        for index, path in enumerate(payload, start=1):
            file.write(f"{sha256_file(path)}  {path.relative_to(release).as_posix()}\n")
            if index % 25 == 0:
                print(f"Checksums: {index}/{len(payload)}", flush=True)
    payload.append(checksum_path)

    local_archive = Path("/content/project-3-submission.zip")
    print("Building submission archive...", flush=True)
    with zipfile.ZipFile(
        local_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3
    ) as archive:
        for index, path in enumerate(payload, start=1):
            archive.write(path, arcname="project-3/" + path.relative_to(release).as_posix())
            if index % 25 == 0:
                print(f"Archived: {index}/{len(payload)}", flush=True)
    print("Verifying archive contents...", flush=True)
    with zipfile.ZipFile(local_archive) as archive:
        damaged = archive.testzip()
        if damaged is not None:
            raise ValueError(f"Archive CRC failure: {damaged}")

    destination = root / local_archive.name
    temporary = destination.with_name(destination.name + ".tmp")
    print("Copying the archive to Google Drive...", flush=True)
    shutil.copyfile(local_archive, temporary)
    archive_hash = sha256_file(local_archive)
    if sha256_file(temporary) != archive_hash:
        raise ValueError("Archive copy checksum mismatch.")
    temporary.replace(destination)
    print("\nSUBMISSION_READY", flush=True)
    print(f"Notebooks: {len(inventory)}/4")
    print(f"Reviewed errors: {len(errors)}")
    print(f"Archive size: {destination.stat().st_size / 2**30:.2f} GiB")
    print(f"SHA256: {archive_hash}")
    print(f"Saved: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

