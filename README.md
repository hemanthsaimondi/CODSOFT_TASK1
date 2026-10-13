# Task 1: Titanic Survival Prediction

Predicts whether a Titanic passenger survived, using the classic Kaggle Titanic dataset (891 passengers).

## Approach
1. **EDA**: survival by sex, class, age, port, fare (`eda.png`)
2. **Feature engineering**: `Title` (from Name), `FamilySize`, `IsAlone`, `HasCabin`
3. **Preprocessing**: median imputation + scaling (numeric); mode imputation + one-hot (categorical), in a single sklearn Pipeline
4. **Models compared** (5-fold CV): Logistic Regression, Random Forest, Gradient Boosting
5. **Tuning**: GridSearchCV on Random Forest
6. **Evaluation**: 80/20 stratified hold-out split

## Results
| Metric | Test set |
|---|---|
| Accuracy | 0.793 |
| Precision | 0.767 |
| Recall | 0.667 |
| F1 | 0.713 |

Best CV accuracy: 0.834. Key drivers: title/sex, ticket class, fare, cabin presence.

## Run
```
pip install pandas numpy scikit-learn matplotlib seaborn
python titanic_survival_prediction.py
```
