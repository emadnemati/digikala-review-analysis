> # تحلیل نظرات دیجی کالا
>
> پروژه طبقه بندی پیشنهاد خرید و بازه امتیاز فقط از متن نظر
>
> مدل پایه `TF-IDF` و مدل منتخب `ParsBERT` آموزش دیده روی ۳ میلیون نظر هستند
>
> [گزارش نهایی](reports/final_report.pdf) شرح روش نتایج و محدودیت ها را دارد
>
> [نسخه متنی گزارش](reports/final_report.md) و [تحلیل ۳۲ خطا](reports/error_analysis.md) نیز موجودند
>
> ## اجرا
>
> کتابخانه ها از `requirements.txt` نصب می شوند
>
> برای اجرای نسخه مخزن پوشه `models` از بسته کامل در ریشه پروژه قرار می گیرد
>
> پیش بینی با فایل های محلی و روی `CPU` اجرا می شود و به بازآموزی نیاز ندارد
>
```bash
python -m pip install -r requirements.txt
python predict.py --text "کیفیتش خوبه ولی قیمتش بالاست." --device cpu
python predict.py --demo --device cpu
```
>
> ## فایل ها
>
> | مسیر | کاربرد |
> | --- | --- |
> | `predict.py` | پیش بینی با مدل پایه و مدل منتخب |
> | `normalize_persian.py` | نرمال سازی مشترک متن |
> | `models/` | مدل ها توکنایزر و تنظیمات پیش بینی |
> | `notebooks/` | چهار نوت بوک تحلیل مدل پایه ترنسفورمر و نیمه نظارتی |
> | `data/` | فایل های تقسیم داده و `split_manifest.json` |
> | `reports/` | گزارش نمودارها معیارها و نمونه های خطا |
> | `artifacts/` | وزن های میانی آزمایش ها در بسته کامل |
>
> نوت بوک ها به ترتیب شماره در `Google Colab` اجرا شده اند
>
> دو نوت بوک اول از `CPU` و دو نوت بوک بعدی از `T4 GPU` استفاده کرده اند
>
> مسیرهای آن ها به پوشه `digikala-project-3` در `Google Drive` وابسته اند
>
> جزئیات بازتولید در پیوست ب گزارش آمده است
>
> گزارش فعلی از لینک بالای این صفحه قابل دریافت است
>
> [داده ها و مدل ها](https://github.com/emadnemati/digikala-review-analysis/releases/latest)
>
> [داده عمومی](https://huggingface.co/datasets/RadeAI/Digikala_comments_products/tree/main) [مدل ParsBERT](https://huggingface.co/HooshvareLab/bert-fa-base-uncased)
