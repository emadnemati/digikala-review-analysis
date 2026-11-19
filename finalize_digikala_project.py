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


def code(value):
    return chr(96) + str(value) + chr(96)


def markdown_table(headers, rows):
    def render(row):
        values = [
            str(value).replace("|", r"\|").replace("\n", " ") for value in row
        ]
        return "| " + " | ".join(values) + " |"

    return "\n".join([
        render(headers), render(["---"] * len(headers)),
        *[render(row) for row in rows],
    ])


def persian_box(text):
    return "\n".join(
        "> " + line if line else ">" for line in text.strip().splitlines()
    )


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


def build_reports(release, results, overlap, schema, error_count):
    reports = release / "reports"
    selected = results["ParsBERT 3M"]
    pseudo = read_json(reports / "pseudo_labeling_1200k.json")
    audit = read_json(reports / "pseudo_label_accuracy_1200k.json")
    comparison = read_json(reports / "semi_supervised_test_comparison.json")
    demo = read_json(reports / "offline_demo.json")
    baseline_training = read_json(reports / "baseline_training.json")
    configuration = read_json(release / "models" / "inference_config.json")

    comparison_rows = []
    for name, result in results.items():
        for target, record in result["heads"].items():
            metrics = record["metrics"]
            comparison_rows.append([
                code(name),
                "پیشنهاد خرید" if target == "recommendation_status" else "بازهٔ امتیاز",
                f"{metrics['accuracy']:.4f}",
                f"{metrics['macro avg']['precision']:.4f}",
                f"{metrics['macro avg']['recall']:.4f}",
                f"{metrics['macro avg']['f1-score']:.4f}",
                f"{record['balanced_accuracy']:.4f}",
            ])
    score_table = markdown_table(
        ["مدل", "هدف", code("Accuracy"), code("Macro Precision"),
         code("Macro Recall"), code("Macro F1"), code("Balanced Accuracy")],
        comparison_rows,
    )
    class_rows = []
    for target, record in selected["heads"].items():
        for label in record["confusion_matrix"]["class_order"]:
            values = record["metrics"][label]
            class_rows.append([
                "پیشنهاد خرید" if target == "recommendation_status" else "بازهٔ امتیاز",
                code(label), f"{values['precision']:.4f}",
                f"{values['recall']:.4f}", f"{values['f1-score']:.4f}",
                f"{int(values['support']):,}",
            ])
    class_table = markdown_table(
        ["هدف", "کلاس", code("Precision"), code("Recall"), code("F1"), "تعداد"],
        class_rows,
    )
    pseudo_table = markdown_table(
        ["کلاس پیش‌بینی‌شده", "پذیرفته‌شده", "درست", "دقت"],
        [[code(label), f"{values['accepted']:,}", f"{values['correct']:,}",
          f"{values['precision']:.2%}" if values["precision"] is not None else "—"]
         for label, values in audit["by_predicted_class"].items()],
    )
    schema_table = markdown_table(
        ["ستون", "نوع استنباط‌شده با بررسی کل فایل"],
        [[code(row["column"]), code(row["inferred_type"])]
         for _, row in schema.iterrows()],
    )
    rec_change = comparison["heads"]["recommendation_status"]["macro_f1"]["change"]
    band_change = comparison["heads"]["rating_band"]["macro_f1"]["change"]
    ordinal_change = comparison["heads"]["rating_band"]["ordinal_distance"]["change"]
    distance = selected["heads"]["rating_band"]["mean_absolute_ordinal_distance"]
    baseline_rec = results["TF-IDF"]["heads"]["recommendation_status"]["metrics"]["macro avg"]["f1-score"]
    selected_rec = selected["heads"]["recommendation_status"]["metrics"]["macro avg"]["f1-score"]

    report = f"""
# گزارش تحلیل نظرهای دیجی‌کالا

## مسئله و داده

ورودی مدل فقط متن نظر در ستون {code('body')} است. دو خروجی داریم، پیشنهاد خرید در سه کلاس
{code('recommended')}، {code('not_recommended')} و {code('no_idea')}،
و بازهٔ امتیاز در سه کلاس {code('low')}، {code('medium')} و {code('high')}.

فایل نظرات ۶٬۱۵۶٬۲۸۹ رکورد و ۱۵ ستون دارد. شمارش با خوانندهٔ {code('CSV')} انجام شد.
۳۳۱٬۵۹۹ محصول و ۲۹٬۲۱۳ کد فروشندهٔ غیرخالی مشاهده شد.
این شمارش‌ها از خروجی بررسی کامل داده در نوت‌بوک اول ثبت شده‌اند.
اندازه و هش فایل منبع در {code('split_manifest.json')} ثبت شده‌اند.

{schema_table}

## یافته‌های تحلیل اولیه

- برچسب پیشنهاد خرید برای ۸۹۴٬۴۲۶ رکورد، معادل ۱۴٫۵۳٪، غایب است.
- این برچسب در ۱۰۰٪ نظرات غیرخریداران و ۱۰٫۱۲٪ نظرات خریداران غایب است.
- از نظرهای دارای برچسب، ۷۹٫۱۱٪ خرید را پیشنهاد می‌کنند. بازهٔ امتیاز بالا ۶۵٫۰۵٪ امتیازهای معتبر را دارد.
- میانهٔ طول متن ۶ واژه و صدک ۹۵ آن ۳۲ واژه است. ۲۷٫۲۲٪ متن‌های غیرخالی حداکثر ۳ واژه دارند.
- میانهٔ تعداد نظر هر محصول ۴ است. یک درصد از محصولات ۱۰٫۸۵٪ نظرات را دارند.
- ۳٬۲۲۹ ردیف اضافه با شناسهٔ تکراری مشاهده شد. ۲۴۱ شناسه چند متن متفاوت داشتند.
- یک امتیاز خارج از دامنه با مقدار ۲۵۰۰ مشاهده شد. دامنهٔ معتبر پروژه صفر تا پنج است.

جدول‌های مقادیر غایب، توزیع‌ها و چهار نمودار در نوت‌بوک اول قرار دارند.
این یافته‌ها مبنای تقسیم محصولی، وزن‌دهی کلاس‌ها، گزارش {code('Macro F1')} و بررسی طول توکن‌ها بودند.

## پاک‌سازی و تقسیم

متن‌های خالی و مقدار متنی {code('nan')} کنار گذاشته شدند.
گروه شناسه‌های دارای چند متن متفاوت حذف شدند.
شناسه‌های تکراری با یک متن ادغام شدند و هدف دارای برچسب‌های معتبر متعارض ماسک شد.
پس از پاک‌سازی ۶٬۱۵۲٬۱۸۲ رکورد باقی ماند.

امتیاز صفر تا کمتر از ۲٫۵ در بازهٔ پایین، ۲٫۵ تا کمتر از ۴ در بازهٔ متوسط
و ۴ تا ۵ در بازهٔ بالا قرار گرفت. کد ترتیبی آن‌ها به‌ترتیب صفر، یک و دو است.

تقسیم با بذر {code('2026')} و هش شناسهٔ محصول انجام شد.
آموزش ۴٬۳۰۷٬۳۱۰، اعتبارسنجی ۹۲۵٬۱۹۷ و آزمون ۹۱۹٬۶۷۵ رکورد دارد.
هر محصول فقط به یک مجموعه تخصیص یافت.

در هر کلاس پیشنهاد خریدِ آموزش، ۲۰٪ با گرد کردن رو به پایین و هش ثابت انتخاب شدند.
بذر ماسک {code('digikala-semi-supervised-v1')} است.
۷۳۶٬۵۴۰ رکورد به مخزن مصنوعی منتقل شدند و فایل آن‌ها فقط {code('id')} و {code('body')} دارد.

فایل آموزش نظارت‌شده ۳٬۵۷۰٬۷۶۵ رکورد دارد.
ردیف دارای یک هدف معتبر برای همان هدف استفاده می‌شود.
نظرهای طبیعیِ بدون برچسب پیشنهاد خرید، در صورت داشتن امتیاز معتبر، در آموزش سر امتیاز هم حضور دارند.

## مدل‌ها و آموزش

مدل پایه {code('TF-IDF')} واژه‌ای با ۵۵٬۳۶۵ ویژگی و رگرسیون لجستیک است.
واژگان و وزن کلاس‌ها از آموزش به دست آمدند.
هر دو مدل پایه در سقف ۶۰ تکرار متوقف شدند و هشدار همگرایی ثبت شد.
نتایج جدول مربوط به همین اجرای ذخیره‌شده است.

مدل عمیق {code('HooshvareLab/bert-fa-base-uncased')} با مجوز {code('Apache-2.0')} است.
مدل دوسر ۱۶۲٬۲۵۵٬۳۶۶ پارامتر دارد و ۷٬۰۹۲٬۴۸۶ پارامتر آن آموزش‌پذیر بود.
بلوک آخر انکودر و دو سر پیش‌بینی تنظیم شدند.

طول ورودی ۹۶ توکن و اندازهٔ دسته ۶۴ است.
در نمونهٔ ثابت ۲۰ هزارنفری آموزش، ۰٫۴۳٪ متن‌ها در این طول بریده می‌شوند.
این نرخ برای کلاس {code('not_recommended')} برابر ۰٫۹۹٪ بود.
نرخ یادگیری انکودر {code('2e-5')} و سرها {code('2e-4')} است.
وزن‌های کلاسِ مدل اولیه در ادامهٔ آموزش هم استفاده شدند.

مدل اولیه سه دور روی ۱۲۰ هزار نظر آموزش دید. سپس یک دور روی ۱٫۲ میلیون نظر
و یک دور دیگر روی ۳ میلیون نظر ادامه یافت.
زیرمجموعه‌ها با هش ثابت از آموزش نظارت‌شده انتخاب شدند.
نمودار آموزش اولیه و گزارش مراحل در پوشهٔ گزارش‌ها قرار دارند.

انتخاب چک‌پوینت بر اساس میانگین {code('Macro F1')} دو هدف روی ۲۰ هزار نظر ثابت اعتبارسنجی انجام شد.
مدل ۳ میلیون‌تایی روی کل اعتبارسنجی نیز ارزیابی شد و سپس ارزیابی آزمون انجام گرفت.

## نتایج آزمون

همهٔ مدل‌های زیر روی همان مجموعهٔ آزمون سنجیده شده‌اند.
برای پیشنهاد خرید ۷۸۵٬۷۷۶ و برای بازهٔ امتیاز ۹۱۹٬۶۷۳ هدف معتبر وجود دارد.

{score_table}

معیارهای هر کلاس برای مدل منتخب:

{class_table}

خطای میانگین فاصلهٔ ترتیبی مدل منتخب برابر {code(f'{distance:.4f}')} است.
ماتریس‌های درهم‌ریختگی در گزارش‌های آزمون {code('JSON')} ذخیره شده‌اند.

## آزمایش نیمه‌نظارتی

مدل والد این آزمایش، مدل نظارت‌شدهٔ ۱٫۲ میلیون‌تایی است.
از {pseudo['scored_rows']:,} نمونهٔ مخزن مصنوعی، {pseudo['accepted_rows']:,} نمونه،
معادل {pseudo['accepted_pct']:.2f}٪، از آستانهٔ ثابت {code('0.90')} گذشتند.

مدل دوم از چک‌پوینت والد شروع کرد و یک دور روی همان ۱٫۲ میلیون نظر نظارت‌شده
به‌علاوهٔ نمونه‌های پذیرفته‌شده آموزش دید.
شبه‌برچسب‌ها فقط در خطای پیشنهاد خرید سهم داشتند.

دقت برچسب‌های پذیرفته‌شده، پس از آموزش مدل دوم، برابر {audit['accuracy']:.2%} محاسبه شد.

{pseudo_table}

تغییر {code('Macro F1')} آزمون برای پیشنهاد خرید {code(f'{rec_change:+.4f}')}
و برای بازهٔ امتیاز {code(f'{band_change:+.4f}')} بود.
تغییر خطای ترتیبی {code(f'{ordinal_change:+.4f}')} است.
بهبود نمره کوچک بود و خطای ترتیبی افزایش یافت.
تفاوت نتیجه شامل ادامهٔ آموزش و افزودن شبه‌برچسب‌هاست.

## محدودیت‌ها و تفسیر

در آزمون، {overlap['test_rows_with_text_in_selected_train']:,} ردیف،
معادل {overlap['test_overlap_pct']:.2f}٪، متنی عیناً یکسان با زیرمجموعهٔ آموزش سه‌میلیونی دارند.
نتیجهٔ آزمون، عملکرد روی محصول‌های جداشده در حضور این هم‌پوشانی متنی را توصیف می‌کند.

متن‌های عمومی مثل «خوب بود» با برچسب‌ها و امتیازهای متفاوت ثبت شده‌اند.
بعضی متن‌های مثبت نیز امتیاز صفر دارند. امتیاز صفر طبق تعریف پروژه در بازهٔ پایین حفظ شد.
کلاس بی‌نظر و امتیاز متوسط همچنان خطای بیشتری دارند.

## پیش‌بینی آفلاین

مدل کامل، توکنایزر، نرمال‌سازی و مدل‌های پایه داخل بسته ذخیره شده‌اند.
اجرای چهار مثال کوتاه، مبهم، منفی و ترکیبی با بارگذاری آفلاین روی {code('CPU')} کولب گذشت.
برای هر هدف، برچسب و احتمال کلاس‌ها برگردانده می‌شود.

زمان چهار پیش‌بینی در همان اجرا به‌ترتیب
{', '.join(str(item['latency_ms']) for item in demo)} میلی‌ثانیه بود.
این اندازه‌گیری چهارنمونه‌ای مربوط به محیط کولب است.
توضیح مدل پایه سهم واقعی واژه‌ها در اختلاف امتیاز کلاس منتخب و کلاس دوم را نشان می‌دهد.

## توضیح ساده برای ارائه

مدل قرار است با خواندن متن نظر حدس بزند کاربر خرید را پیشنهاد می‌کند یا نه و چه بازهٔ امتیازی داده است.
ابتدا داده‌ها را بررسی کردیم و تکرارهای مسئله‌دار را پاک کردیم.
نظرهای هر محصول را در یک مجموعه نگه داشتیم تا مدل روی محصول‌های جداشده سنجیده شود.

اول مدل پایه را ساختیم و بعد مدل فارسی ازپیش‌آموزش‌دیده را روی نظرهای دیجی‌کالا تنظیم کردیم.
نمرهٔ پیشنهاد خرید مدل پایه {code(f'{baseline_rec:.4f}')} و مدل منتخب {code(f'{selected_rec:.4f}')} شد.

در آزمایش نیمه‌نظارتی، جواب بخشی از دادهٔ آموزش را موقتاً کنار گذاشتیم.
مدل برای این بخش حدس زد و حدس‌های با احتمال حداقل ۹۰٪ را به آموزش اضافه کردیم.
بهبود نهایی این آزمایش کوچک بود. مدل منتخب برای تحویل، مدل سه‌میلیونی است.
"""
    error_text = (reports / "error_analysis.md").read_text(encoding="utf-8")
    error_text = error_text.replace(
        "> # تحلیل ۲۴ خطای مدل منتخب", "> # تحلیل ۳۲ خطای مدل منتخب", 1
    )
    (reports / "error_analysis.md").write_text(error_text, encoding="utf-8")
    unquoted = "\n".join(
        line[2:] if line.startswith("> ") else "" if line == ">" else line
        for line in error_text.splitlines()
    )
    report += "\n\n" + unquoted
    (reports / "final_report.md").write_text(
        "![Test macro F1](figures/test_macro_f1.png)\n\n"
        + persian_box(report) + "\n", encoding="utf-8"
    )

    readme = f"""
# تحلیل نظرهای دیجی‌کالا

این پروژه فقط از متن نظر، پیشنهاد خرید و بازهٔ امتیاز را پیش‌بینی می‌کند.
مدل منتخب {code('ParsBERT')} سه‌میلیونی است. اسکریپت هر دو مدل پایه و منتخب را اجرا می‌کند.

ابتدا کتابخانه‌های {code('requirements.txt')} را نصب کن.
سپس {code('predict.py')} را با متن نظر اجرا کن.
گزینهٔ {code('--demo')} چهار مثال آماده را اجرا می‌کند.
مدل از فایل‌های داخل پوشهٔ {code('models')} بارگذاری می‌شود.
نسخهٔ پایتون محیط ثبت‌شده {code(configuration['python_version'])} است.

پوشهٔ {code('reports')} گزارش نهایی، تحلیل {error_count} خطا، معیارها، نمودار و مثال‌های اجرا را دارد.
پوشهٔ {code('data')} پنج فایل تقسیم و {code('split_manifest.json')} را دارد.
پوشهٔ {code('artifacts')} چک‌پوینت‌های آزمایش‌ها را نگه می‌دارد.

نوت‌بوک‌ها به این ترتیب هستند:

۱. {code(NOTEBOOKS[0])} برای تحلیل و آماده‌سازی، با {code('CPU')}
۲. {code(NOTEBOOKS[1])} برای مدل پایه، با {code('CPU')}
۳. {code(NOTEBOOKS[2])} برای تنظیم مدل، با {code('T4 GPU')}
۴. {code(NOTEBOOKS[3])} برای نیمه‌نظارتی، ارزیابی و تحویل، با {code('T4 GPU')}

نوت‌بوک‌ها برای مسیر موقت {code('/content')} و پوشهٔ {code('digikala-project-3')} در
{code('Google Drive')} نوشته شده‌اند.
برای بازتولید در کولب، محتوای بسته را در همین پوشهٔ درایو قرار بده و کتابخانه‌های ثبت‌شده را نصب کن.

نوت‌بوک‌ها خروجی‌های اجراهای تعاملی این پروژه را دارند.
اجرای یکپارچهٔ هر چهار نوت‌بوک از صفر در این مرحله بررسی نشده است.
آزمون تأییدشدهٔ بسته، اجرای آفلاین اسکریپت روی چهار مثال با {code('CPU')} است.
"""
    fence = chr(96) * 3
    commands = (
        f"\n\n{fence}bash\n"
        "pip install -r requirements.txt\n"
        'python predict.py --text "کیفیتش خوبه ولی قیمتش بالاست." --device cpu\n'
        "python predict.py --demo --device cpu\n"
        f"{fence}\n"
    )
    (release / "README.md").write_text(
        persian_box(readme) + commands, encoding="utf-8"
    )
    return baseline_training, comparison, audit


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

    inventory = collect_notebooks(root, release)
    figures = release / "reports" / "figures"
    draw_comparison(results, figures / "test_macro_f1.png")
    baseline_training, comparison, audit = build_reports(
        release, results, overlap, schema, len(errors)
    )

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
        "reports/final_report.md", "reports/error_analysis.md",
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

