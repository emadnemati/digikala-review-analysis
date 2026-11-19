> # تحلیل نظرهای دیجی‌کالا
>
> این پروژه فقط از متن نظر، پیشنهاد خرید و بازهٔ امتیاز را پیش‌بینی می‌کند.
> مدل منتخب `ParsBERT` سه‌میلیونی است. اسکریپت هر دو مدل پایه و منتخب را اجرا می‌کند.
>
> ابتدا کتابخانه‌های `requirements.txt` را نصب کن.
> سپس `predict.py` را با متن نظر اجرا کن.
> گزینهٔ `--demo` چهار مثال آماده را اجرا می‌کند.
> مدل از فایل‌های داخل پوشهٔ `models` بارگذاری می‌شود.
> نسخهٔ پایتون محیط ثبت‌شده `3.13.16` است.
>
> پوشهٔ `reports` گزارش نهایی، تحلیل 32 خطا، معیارها، نمودار و مثال‌های اجرا را دارد.
> پوشهٔ `data` پنج فایل تقسیم و `split_manifest.json` را دارد.
> پوشهٔ `artifacts` چک‌پوینت‌های آزمایش‌ها را نگه می‌دارد.
>
> نوت‌بوک‌ها به این ترتیب هستند:
>
> ۱. `01_descriptive_analysis.ipynb` برای تحلیل و آماده‌سازی، با `CPU`
> ۲. `02_baselines.ipynb` برای مدل پایه، با `CPU`
> ۳. `03_transfer_learning_finetuning.ipynb` برای تنظیم مدل، با `T4 GPU`
> ۴. `04_semi_supervised.ipynb` برای نیمه‌نظارتی، ارزیابی و تحویل، با `T4 GPU`
>
> نوت‌بوک‌ها برای مسیر موقت `/content` و پوشهٔ `digikala-project-3` در
> `Google Drive` نوشته شده‌اند.
> برای بازتولید در کولب، محتوای بسته را در همین پوشهٔ درایو قرار بده و کتابخانه‌های ثبت‌شده را نصب کن.
>
> نوت‌بوک‌ها خروجی‌های اجراهای تعاملی این پروژه را دارند.
> اجرای یکپارچهٔ هر چهار نوت‌بوک از صفر در این مرحله بررسی نشده است.
> آزمون تأییدشدهٔ بسته، اجرای آفلاین اسکریپت روی چهار مثال با `CPU` است.
>
>
> ## نسخهٔ مخزن و بستهٔ کامل
>
> کد، نوت‌بوک‌ها و گزارش‌ها در مخزن `Git` نگهداری می‌شوند.
> داده‌ها، پایگاه داده و وزن‌های مدل داخل بستهٔ کامل `project-3-submission.zip` قرار دارند.
> برای اجرای پیش‌بینی از یک کلون تازه، پوشهٔ `models` داخل `project-3` در آرشیو را به ریشهٔ مخزن کپی کن.
> فایل `checksums.sha256` مربوط به محتویات بستهٔ کامل تحویل است.

```bash
pip install -r requirements.txt
python predict.py --text "کیفیتش خوبه ولی قیمتش بالاست." --device cpu
python predict.py --demo --device cpu
```

[Complete submission and model files](https://github.com/emadnemati/digikala-review-analysis/releases)

[Public dataset](https://huggingface.co/datasets/RadeAI/Digikala_comments_products/tree/main)

[ParsBERT model card and license](https://huggingface.co/HooshvareLab/bert-fa-base-uncased)
