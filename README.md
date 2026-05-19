# Craiglist-cars-price-prediction
Repo for SPBU Technologies of AI project

__Dataset:__ [Used Cars Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data?utm_source=chatgpt.com)

## Project Structure

```
project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_features.ipynb
│   ├── 05_training.ipynb
│   └── 06_error_analysis.ipynb
│
├── src/
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── reports/
│   ├── design_document.md
│   └── figures/
│
├── models/
│   └── final_model.pkl
│
└── app/
    └── app.py
```
