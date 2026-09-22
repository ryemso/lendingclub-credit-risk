# LendingClub Credit Risk Modeling

LendingClub 대출 데이터를 이용해 **상환 실패(Default / Charged Off) 가능성을 분류**하고, 클래스 불균형 처리와 여러 분류 모델을 비교한 프로젝트입니다.

이 저장소는 Google Drive에 보관된 원본 Colab notebook `9_Project_code.ipynb`의 분석 흐름을 기준으로 재구성했습니다.

## Problem Definition

원본 notebook에서는 다음 loan status만 사용했습니다.

- `Fully Paid` → 0
- `Charged Off`, `Default` → 1

즉 **상환 성공 vs 상환 실패**의 이진 분류 문제로 구성했습니다.

## Selected Features

원본 notebook에서 사용한 주요 변수:

- loan amount / term / interest rate / installment
- grade / sub_grade
- annual income / DTI
- FICO range
- open accounts / total accounts
- revolving balance / utilization
- bankruptcy / delinquency / recent inquiries
- home ownership / verification status / purpose

## Preprocessing

원본 notebook에서 확인되는 주요 처리:

- `emp_title` 제거
- DTI 결측치 조건부 처리
- `inq_last_6mths` 결측 0 대체
- `revol_util` 결측을 `revol_bal` 조건에 따라 처리
- bankruptcy 결측 행 제거
- `sub_grade` ordinal encoding
- home ownership / purpose / verification status encoding
- `fico_avg = (fico_range_low + fico_range_high) / 2`
- `installment_to_loan`, `acc_ratio` 파생변수
- 일부 long-tail 변수 `log1p` 변환
- train/test stratified split
- train 기준 StandardScaler fit 후 test transform

## Imbalanced Data Experiments

원본 notebook에는 다음 방식이 실제로 비교되어 있습니다.

### Oversampling
- SMOTE
- ADASYN
- Borderline-SMOTE

### Undersampling / cleaning
- RandomUnderSampler
- NearMiss
- Tomek Links

## Model Comparison

원본 notebook에서 비교한 모델:

- KNN
- Logistic Regression
- Decision Tree
- Random Forest
- SVM (RBF)
- XGBoost
- LightGBM

평가 지표:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

XGBoost에서는 추가로:

- RandomizedSearchCV
- Threshold sweep
- ROC curve
- Precision–Recall curve
- KS statistic
- Confusion Matrix
- Feature Importance
- SHAP
- Learning Curve

까지 확인했습니다.

## Recorded Results

원본 notebook에 정리된 tuned model table 기준:

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.6025 | 0.2947 | 0.7112 | 0.4167 | 0.7017 |
| XGBoost | 0.6430 | 0.3167 | 0.6805 | **0.4322** | **0.7163** |
| Logistic Regression | 0.6547 | 0.3193 | 0.6445 | 0.4270 | 0.7065 |
| Decision Tree | 0.5677 | 0.2816 | **0.7509** | 0.4096 | 0.6858 |
| LightGBM | 0.7712 | 0.4022 | 0.2966 | 0.3434 | 0.7006 |

한 가지 지표만으로 모델을 고르기보다 Recall, F1, ROC AUC의 trade-off를 함께 비교한 프로젝트입니다.

원본 notebook의 Logistic Regression RandomizedSearch 결과:

- Accuracy: **0.6543**
- Precision: **0.3183**
- Recall: **0.6406**
- F1: **0.4253**
- ROC AUC: **0.7049**

## Reviewable Code

[src/modeling_pipeline.py](./src/modeling_pipeline.py)는 원본 notebook의 분석 흐름을 면접/코드리뷰용으로 재구성한 파일입니다.

- 개인 Google Drive 경로 제거
- 데이터 파일 비공개
- 함수 단위로 전처리/학습/평가 구조화
- 원본에서 사용한 feature engineering, SMOTE/RUS, model comparison 흐름 유지

원본 notebook 전체를 그대로 복사한 것이 아니라 **검토 가능한 형태로 재구성한 버전**입니다.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
└── src/
    └── modeling_pipeline.py
```

## Run

```bash
pip install -r requirements.txt
python src/modeling_pipeline.py
```

원본 LendingClub CSV는 저장소에 포함하지 않습니다.

```text
data/accepted_2007_to_2018Q4.csv
```

## Tech Stack

**Python · Pandas · Scikit-learn · Imbalanced-learn · XGBoost · LightGBM · SHAP**
