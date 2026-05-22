# Design Document: Used Car Price Prediction

## 1. Goal and Metrics

### 1.1. Goal

**Project title:** Used Car Price Prediction

**Task:** предсказание цены подержанного автомобиля по характеристикам объявления.

**Task type:** regression

**Target:** `price`

**Dataset:** Used Cars Dataset, Kaggle, Craigslist Cars and Trucks Data

**Goal:**
Цель проекта — построить модель, которая по валидным характеристикам автомобиля и объявления предсказывает цену подержанного автомобиля.

---

### 1.2. Why This Task Is Important

Модель может помочь продавцам оценивать адекватную цену автомобиля, а покупателям — находить объявления с завышенной или заниженной ценой. Подобные технологии уже активно используются в реальных примерах различных сервисов.

---

### 1.3. Metrics

Основная метрика: `MAE`

MAE показывает среднюю абсолютную ошибку модели в долларах.

```text
MAE = mean(abs(y_true - y_pred))
```

Почему используется MAE:

```text
MAE выбрана основной метрикой, потому что она измеряет среднюю абсолютную ошибку в тех же единицах, что и целевая переменная, то есть в долларах. Это делает результат легко интерпретируемым: MAE показывает, на сколько долларов модель в среднем ошибается при предсказании цены автомобиля.

Для данной задачи это особенно удобно, так как цель проекта — оценивать цену подержанного автомобиля. Кроме того, в данных есть выбросы и нетипичные цены, а MAE менее чувствительна к отдельным экстремальным ошибкам, чем RMSE. Поэтому MAE лучше отражает типичную ошибку модели и используется как основная метрика.
```

---

### 1.4. Success Criteria

Модель считается успешной, если:

```text
- MAE модели ниже MAE baseline;
- качество стабильно на валидации/тесте;
- модель не показывает явного переобучения;
- ошибки модели интерпретируемы.
```

---

## 2. Data

### 2.1. Dataset Source

**Dataset name:** Used Cars Dataset

**Source:** Kaggle

**Original platform:** Craigslist

**Link:**
[https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)

---

### 2.2. Dataset Size

Initial dataset:

```text
Rows: 426880
Columns: 26
```

After cleaning:

```text
Rows: 270851
Columns: 19
```

---

### 2.3. Target Variable

Target variable:

```text
price
```

Description:

```text
Цена автомобиля в долларах США, указанная в объявлении.
```

Problems with target:

```text
- есть нулевые цены;
- есть слишком маленькие цены;
- есть экстремально большие цены;
- распределение price скошено вправо.
```

---

### 2.4. Feature Description

| Feature        | Type             | Description           | Use / Drop    |
| -------------- | ---------------- | --------------------- | ------------- |
| `id`           | numeric          | ID объявления         | drop          |
| `url`          | text             | ссылка на объявление  | drop          |
| `region`       | categorical      | регион                | use           |
| `region_url`   | text             | ссылка на регион      | drop          |
| `price`        | numeric          | цена автомобиля       | target        |
| `year`         | numeric          | год выпуска           | use           |
| `manufacturer` | categorical      | производитель         | use           |
| `model`        | categorical/text | модель автомобиля     | use           |
| `condition`    | categorical      | состояние             | use           |
| `cylinders`    | categorical      | количество цилиндров  | use           |
| `fuel`         | categorical      | тип топлива           | use           |
| `odometer`     | numeric          | пробег                | use           |
| `title_status` | categorical      | юридический статус    | use           |
| `transmission` | categorical      | коробка передач       | use           |
| `VIN`          | text             | VIN автомобиля        | drop          |
| `drive`        | categorical      | тип привода           | use           |
| `size`         | categorical      | размер автомобиля     | drop          |
| `type`         | categorical      | тип кузова            | use           |
| `paint_color`  | categorical      | цвет                  | use           |
| `image_url`    | text             | ссылка на изображение | drop          |
| `description`  | text             | описание объявления   | use           |
| `county`       | unknown          | округ                 | drop          |
| `state`        | categorical      | штат                  | use           |
| `lat`          | numeric          | широта                | use           |
| `long`         | numeric          | долгота               | use           |
| `posting_date` | datetime         | дата публикации       | use           |

---

### 2.5. Features Planned for Removal

| Feature      | Reason                                             |
| ------------ | -------------------------------------------------- |
| `id`         | технический идентификатор объявления               |
| `url`        | ссылка на объявление, не характеристика автомобиля |
| `region_url` | техническая ссылка региона                         |
| `image_url`  | ссылка на изображение, изображения не используются |
| `VIN`        | слишком специфичный идентификатор                  |
| `county`     | пропущено значение у всех элементов датасета       |
| `size`       | > 70% пропусков                                    |

---

### 2.6. Non-Triviality of Dataset

Датасет не является тривиальным, потому что:

```text
- много пропусков;
- есть выбросы в price, year, odometer;
- много категориальных признаков;
- есть текстовый признак description;
- есть временной признак posting_date;
- есть географические признаки: lat, long, region, state;
- некоторые признаки требуют ручной обработки, например, cylinders.
```

---

### 2.7. Data Limitations

Основные ограничения данных:

```text
- цена в объявлении не всегда равна реальной цене продажи;
- данные относятся только к Craigslist;
- объявления могут содержать ошибки;
- в датасете нет информации об авариях, комплектации и истории обслуживания;
- рынок автомобилей меняется со временем.
```

---

## 3. EDA

### 3.1. General Dataset Overview

Main observations:

```text
После очистки и разбиения train-выборка содержит 189596 строк и 19 признаков.

Наибольшие пропуски в train: cylinders (37.806705%), condition (35.470685%), drive (29.083947%), paint_color (27.659339%), type (22.938775%).

В train целевая переменная уже ограничена диапазоном 500..100000, но распределение остается правосторонне скошенным (mean 17860.46 > median 13995.00).
```

---

### 3.2. Missing Values

| Feature        | Missing values, % | Decision               |
| -------------- | ----------------: | ---------------------- |
| `year`         |           0.280069 | заполнить модой        |
| `manufacturer` |           4.330260 | заполнить `unknown`    |
| `model`        |           1.356569 | заполнить `unknown`, редкие модели -> `other` |
| `condition`    |          35.470685 | заполнить `unknown`    |
| `cylinders`    |          37.806705 | перевести текст в число, пропуски -> `0` |
| `fuel`         |           0.564885 | заполнить `unknown`    |
| `odometer`     |           0.430916 | заполнить средним      |
| `title_status` |           1.412477 | заполнить `unknown`    |
| `transmission` |           0.416148 | заполнить `unknown`    |
| `drive`        |          29.083947 | заполнить `unknown`    |
| `size`         |          71.767476 | удалить колонку        |
| `type`         |          22.938775 | заполнить `unknown`    |
| `paint_color`  |          27.659339 | заполнить `unknown`, затем преобразовать в ценовой кластер и удалить исходный признак |
| `description`  |           0.000527 | заполнить пустой строкой |
| `lat`          |           0.645056 | заполнить медианой по штату, затем глобальной медианой |
| `long`         |           0.645056 | заполнить медианой по штату, затем глобальной медианой |
| `posting_date` |           0.000000 | преобразовать в datetime, проверить производные признаки и удалить |
| `region`       |           0.000000 | преобразовать в `region_price_cluster`, исходный признак удалить |
| `state`        |           0.000000 | использовать для импутации `lat/long`, затем удалить |

Conclusion:

```text
Основные пропуски сосредоточены в категориальных признаках; для них применяется заполнение `unknown`, для числовых признаков — статистическая импутация, а для геокоординат — заполнение медианами по штату с резервом на глобальную медиану.
```

---

### 3.3. Target Analysis: `price`

Checked:

```text
Minimum price: 500.00
Maximum price: 100000.00
Mean price: 17860.46
Median price: 13995.00
Number of zero prices: 0
Number of suspicious prices: 0 (цены < 500 или > 100000)
```

Main observations:

```text
Распределение `price` правосторонне скошено, среднее заметно выше медианы.
```

Decision:

```text
Целевую переменную оставляем в очищенном диапазоне [500, 100000]; для анализа дополнительно используем `log_price`.
```

---

### 3.4. Numerical Features Analysis

Analyzed features:

```text
year
odometer
lat
long
```

#### 3.4.1. `year`

Observations:

```text
До фильтрации минимальный `year` = 1900, максимальный = 2022; в train 1482 объявлений с `year < 1960`.
```

Decision:

```text
Оставлены только объявления с `year >= 1980`; пропуски в `year` заполняются модой.
```

#### 3.4.2. `odometer`

Observations:

```text
После фильтрации по `year` наблюдаются выбросы в `odometer`: min = 0, max = 10000000, `odometer = 0` встречается 362 раза, `odometer > 500000` — 525 раз.
```

Decision:

```text
Оставлены записи с `0 < odometer <= 500000`, пропуски заполняются средним, дополнительно создается `log_odometer`.
```

#### 3.4.3. `lat`, `long`

Observations:

```text
Доля пропусков в `lat` и `long` по 0.645056%; координаты хорошо предсказывают `state` (accuracy 0.8616315367827841), но слабо предсказывают `region` (accuracy 0.287688676656238).
```

Decision:

```text
Добавлен признак `lat_long_missing`; `lat/long` заполняются медианой по `state`, затем глобальной медианой. После этого `state` удаляется.
```

---

### 3.5. Categorical Features Analysis

Analyzed features:

```text
manufacturer
model
condition
cylinders
fuel
title_status
transmission
drive
size
type
paint_color
state
region
```

Main observations:

```text
Высокая доля пропусков в категориальных признаках: `cylinders`, `condition`, `drive`, `paint_color`, `type`.

`model` — признак очень высокой кардинальности: 20445 уникальных значений (с учетом NaN), при этом 19067 встречаются реже 20 раз.

`region` содержит 404 уникальных значения и не всегда однозначно соответствует `state` (например, `jackson` встречается в 3 штатах).
```

Decision:

```text
Для `type`, `drive`, `transmission`, `fuel`, `condition`, `manufacturer`, `title_status` пропуски заполняются `unknown`; для `model` редкие категории объединяются в `other`; для `cylinders` применяется маппинг в числовой формат с отдельным флагом `cylinders_other`.
```

---

### 3.6. High-Cardinality Features

High-cardinality features:

```text
description
model
region
```

Problem:

```text
`description`: 174065 уникальных значений, почти каждое описание уникально.
`model`: 20444 уникальных значения (без NaN), много редких категорий.
`region`: 404 категории, высокая размерность при one-hot кодировании.
```

Decision:

```text
`description`: TF-IDF (`max_features=20000`, `ngram_range=(1, 2)`, `min_df=5`).
`model`: пропуски -> `unknown`, редкие модели (<20) объединяются в `other`.
`region`: преобразование в `region_price_cluster` (KMeans по медианной цене, 10 кластеров).
```

---

### 3.7. Text Feature: `description`

Decision:

```text
Пропуски в `description` заполняются пустой строкой для обработки на следующих этапах.
```

---

### 3.8. Date Feature: `posting_date`

Observations:

```text
После фильтров `posting_year` содержит только 1 уникальное значение, `posting_month` — 2 уникальных значения.
```

Decision:

```text
`posting_date` преобразуется в datetime, временные признаки проверяются и удаляются из-за низкой вариативности.
```

Possible extracted features:

```text
posting_year
posting_month
posting_dayofweek
is_weekend
```

---

### 3.9. Feature-Target Relationships

Checked relationships:

```text
price vs year
price vs odometer
price by manufacturer
price by condition
price by fuel
price by transmission
price by drive
price by type
price by state
region by state
state by lat, long
region by lat, long
```

Main observations:

```text
На графике `price` vs `year` более новые автомобили в среднем стоят дороже.

На графике `price` vs `odometer` наблюдается отрицательная связь: с ростом пробега цена в среднем ниже.

`lat/long` хорошо несут информацию о `state` (accuracy 0.8616), а для `region` сигнал заметно слабее (accuracy 0.2877), поэтому `region` преобразуется в ценовые кластеры.
```

---

### 3.10. EDA Summary

Main EDA conclusions:

```text
1. Для `year` и `odometer` применена фильтрация выбросов: `year >= 1980`, `0 < odometer <= 500000`.

2. Пропуски в категориальных признаках обрабатываются через `unknown`, в числовых — статистической импутацией.

3. Высококардинальные признаки обрабатываются специальными методами: `description` -> TF-IDF, `model` -> объединение редких категорий, `region` -> ценовые кластеры.

4. Геопризнаки `lat/long` сохраняются, добавляется `lat_long_missing`, признак `state` удаляется после использования для импутации.

5. После EDA формируется `train_filtered.csv` (182190 строк, 19 колонок) для обучения и кросс-валидации.
```

---

## 4. Validation

### 4.1. Validation Strategy

Chosen validation strategy:

```text
Разделение train/test (Hold-out) + кросс-валидация (K-Fold) на обучающей выборке.
```

---

### 4.2. Split Proportions

```text
Обучающая выборка (Train): 70%
Тестовая выборка (Test): 30%
Cross-validation на Train: 3 или 5 фолдов (KFold)
```

---

### 4.3. Reason for This Split

```text
Разделение 70/30 оставляет достаточный объем данных для обучения сложных ансамблевых моделей и при этом формирует репрезентативный, независимый отложенный набор (Test) для финальной оценки обобщающей способности. На обучающей выборке применяется K-Fold кросс-валидация для устойчивой и объективной оценки качества при подборе гиперпараметров (GridSearchCV).
```

---

### 4.4. Test Set Usage

```text
Тестовый набор используется строго один раз — для оценки финальной версии пайплайна (StackingRegressor).
```

Reason:

```text
Однократное использование тестового набора гарантирует отсутствие переобучения под тест (test set overfitting) и позволяет получить честную и несмещенную оценку метрики MAE, которую модель покажет в реальных (production) условиях.
```

---

### 4.5. Data Leakage Prevention

To avoid data leakage:

```text
1. Все этапы предобработки (импьютация средними/модами, кластеризация регионов с KMeans, вычисление TF-IDF словарей) инкапсулированы в scikit-learn Pipeline.
2. Пайплайн обучается (методом fit) исключительно на обучающих фолдах.
3. Валидационная и тестовая выборки подвергаются только трансформации (методом transform).
4. Оценка и выбор базовых моделей производятся исключительно на кросс-валидации обучающей выборки (без использования отложенного Test-сета).
```

---

### 4.6. Validation Metrics

Metrics used on cross-validation:

```text
Основные метрики: MAE (Mean Absolute Error) и MAPE (Mean Absolute Percentage Error).
Для GridSearchCV используется: neg_mean_absolute_error.
```

---

## 5. Approach

### 5.1. General Approach

Main pipeline:

```text
1. Загрузить данные.
2. Очистить некорректные значения.
3. Удалить технические колонки и полные дубликаты строк.
4. Разбить данные на train/test (70/30).
5. Провести EDA на train и применить фильтры.
6. Построить baseline и оценить по кросс валидации.
7. Провести Feature Engineering (генерация новых признаков, кластеризация).
8. Сформировать общий Pipeline трансформации.
9. Обучить базовые и продвинутые ML-модели, подобрать гиперпараметры.
10. Объединить лучшие алгоритмы в ансамбль (Stacking) и провести финальный Error Analysis.
```

---

### 5.2. Data Cleaning Plan

| Problem                    | Decision               |
| -------------------------- | ---------------------- |
| zero or very low `price`   | удалить: оставить `price >= 500` |
| extremely high `price`     | удалить: оставить `price <= 100000` |
| incorrect `year`           | фильтр: оставить `year >= 1980` |
| extreme `odometer`         | фильтр: оставить `0 < odometer <= 500000` |
| missing categorical values | заполнить `unknown`    |
| missing `description`      | заполнить пустой строкой |
| missing `lat`, `long`      | заполнить медианой по штату, затем глобальной медианой |
| technical columns          | удалить                |
| `cylinders` as text        | перевести в числовой формат |
| low-variance date features | извлечь из `posting_date`, затем удалить |

---

### 5.3. Feature Engineering Plan

Planned new features:

| New Feature              | Description                        |
| ------------------------ | ---------------------------------- |
| `car_age`                | возраст автомобиля (`2022 - year`) |
| `log_odometer`           | `log1p(odometer)` после фильтрации и импутации |
| `lat_long_missing`       | флаг отсутствия хотя бы одной координаты |
| `region_price_cluster`   | кластер региона по медианной цене (KMeans, 10 кластеров) |
| `paint_color_price_cluster` | кластер цвета по медианной цене (KMeans, 5 кластеров) |
| `cylinders_other`        | флаг категории `other` в исходном `cylinders` |

---

### 5.4. Numerical Features

Numerical features:

```text
year
car_age
cylinders
odometer
lat
long
log_odometer
lat_long_missing
cylinders_other
```

Processing:

```text
`year` заполняется модой, `odometer` — средним; `lat/long` заполняются медианой по штату и затем глобальной медианой. Для `odometer` строится `log_odometer`, для пропусков координат — `lat_long_missing`.
```

---

### 5.5. Categorical Features

Categorical features:

```text
manufacturer
model
condition
fuel
title_status
transmission
drive
type
region_price_cluster
paint_color_price_cluster
description
```

Processing:

```text
Для `manufacturer`, `condition`, `fuel`, `title_status`, `transmission`, `drive`, `type` пропуски заменяются на `unknown`. Для `model` редкие значения (менее 20 вхождений) объединяются в `other`. Категориальные признаки кодируются через One-Hot Encoder, `description` кодируется через TF-IDF.
```

---

### 5.6. High-Cardinality Features

High-cardinality features:

```text
description
model
region
```

Processing:

```text
`description`: TF-IDF (`max_features=500`, `ngram_range=(1, 1)`, `min_df=10`).
`model`: оставить популярные модели, остальные -> `other`, пропуски -> `unknown`.
`region`: преобразовать в `region_price_cluster` с помощью KMeans (10 кластеров) по медианной цене региона.
```

---

### 5.7. Preprocessing Pipeline

Planned preprocessing:

```text
1. Применить кастомный трансформер `CraigslistFeatureEngineer`:
   - заполнить пропуски для числовых и категориальных признаков;
   - рассчитать `car_age`, `log_odometer` и другие новые фичи;
   - объединить редкие категории `model`;
   - кластеризовать `region` и `paint_color` в ценовые кластеры через KMeans;
   - удалить исходные избыточные столбцы.
2. Закодировать признаки через `ColumnTransformer`:
   - `description` -> TF-IDF Vectorizer;
   - категориальные признаки -> OneHotEncoder(handle_unknown="ignore");
   - числовые -> passthrough.
```

Important rule:

```text
Все шаги preprocessing обучаются (fit) только на train-данных.
Validation и test преобразуются (transform) уже обученным пайплайном, что исключает Data Leakage.
```

---

### 5.8. Baseline Model

Базовые модели сравнивались на `train_filtered.csv` с 5-fold CV по MAE:

```text
DummyRegressor (median):  MAE = 10657.99
LinearRegression:         MAE =  6192.72
```

Лучшая baseline-модель: `LinearRegression`.

---

### 5.9. Training

Training architecture and pipeline:

```text
Процесс обучения выстроен вокруг единого и неделимого пайплайна (sklearn.pipeline.Pipeline), включающего в себя весь Feature Engineering, Encoding и итоговый Estimator.

Особое внимание уделено целевой переменной (Target):
Из-за сильной правосторонней скошенности цен алгоритмам сложно штрафовать крупные выбросы. Поэтому обучение инкапсулировано в `TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)`.
Модели учатся предсказывать логарифм цены, что нормализует распределение остатков, а на этапе инференса (predict) пайплайн автоматически применяет экспоненту, возвращая предсказание и считая MAE в исходных долларах.
```

---

### 5.10. Model Selection

Tested models and hyperparameter tuning:

```text
Отбор лучшей модели производился с использованием GridSearchCV на обучающей выборке. Для оптимизации времени экспериментов (в рамках академического проекта) проверялись компактные, но репрезентативные сетки гиперпараметров.

1. Decision Tree:
   - Быстро переобучается на признаках после OHE.
   - Поиск велся по `max_depth` и `min_samples_split`.
   - Результат на CV: MAE ~6800. Наихудшая обобщающая способность.

2. Random Forest:
   - Устойчив к переобучению благодаря бэггингу (композиции независимых деревьев).
   - Результат на CV: MAE ~5408. Отличный бейзлайн среди сложных алгоритмов.

3. XGBoost:
   - Продвинутая реализация классического градиентного бустинга.
   - Показал высокую скорость обучения и отличную метрику.
   - Результат на CV: MAE ~5028.

4. CatBoost:
   - Градиентный бустинг, изначально оптимизированный под работу с категориальными переменными, который также оказался крайне эффективен для разреженных матриц (TF-IDF + OHE).
   - Поиск гиперпараметров: `iterations`, `depth`.
   - Результат на CV: MAE ~4908.
```

Final Ensembling (Stacking):

```text
Для финального решения был разработан ансамбль (StackingRegressor), призванный извлечь максимальную пользу из всех мощных алгоритмов одновременно.
- Уровень 0 (Base Estimators): Лучшие версии моделей Random Forest, XGBoost и CatBoost, настроенные с помощью GridSearchCV.
- Уровень 1 (Meta Estimator): Алгоритм Ridge (Линейная регрессия с L2 регуляризацией). Он обучается комбинировать предсказания леса и бустингов так, чтобы минимизировать итоговую ошибку (присваивая им оптимальные веса).
- Финальный результат ансамбля: MAE ~4745.

Именно StackingRegressor стал финальной архитектурой и был интегрирован в итоговый файл (`final_model.pkl`).
```

---

### 5.11. Expected Challenges

```text
Высокая кардинальность `description` и `model` сильно увеличивает размерность признакового пространства, что усложняет использование алгоритмов, чувствительных к разреженным матрицам (например, простых деревьев).
Также, специфика ценообразования на рынке б/у автомобилей содержит скрытые факторы (коллекционная ценность раритетных машин, кастомный тюнинг), которые невозможно предсказать исключительно по табличным данным, что формирует несжимаемый порог ошибки для моделей.
```

---

### 5.12. Approach Summary

```text
В рамках проекта был проведен полноценный цикл ML-разработки:
1. Очистка данных и фильтрация аномалий (по году, пробегу и стоимости).
2. Разведочный анализ (EDA) для понимания зависимостей между пробегом, возрастом, характеристиками и рыночной ценой автомобиля.
3. Feature Engineering, включая сложную кластеризацию регионов по ценовым профилям и внедрение методов NLP (TF-IDF) для текстов объявлений.
4. Проектирование Training & Validation архитектуры: внедрен единый кросс-валидационный пайплайн, защищенный от утечек данных (Data Leakage) и оборачивающий таргет в логарифм (TransformedTargetRegressor).
5. Model Selection: Протестированы классические алгоритмы (Decision Tree, Random Forest) и передовые градиентные бустинги (CatBoost, XGBoost). Финальным решением стал мощный ансамбль StackingRegressor.
6. Выполнен Error Analysis на основе перестановочной важности (Permutation Feature Importance), подтвердивший адекватность модели и высокую значимость таких признаков, как возраст авто, пробег и марка производителя.
```
