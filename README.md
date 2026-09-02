# Employee Attrition Prediction

## Business Problem
Employee attrition costs companies significant money in 
recruitment and training. This project builds a machine 
learning model to predict which employees are likely to 
leave, enabling HR teams to intervene proactively.

## Dataset
- Source: IBM HR Analytics Employee Attrition Dataset
- 1470 employees, 13 features
- Class imbalance: 83.8% stayed, 16.2% left

## Key EDA Findings
1. Employees who left earned 30% less on average (4787 vs 6832)
2. Quitters live further from office on average (10.6 vs 8.9 )
3. Lower job and enviromental satisfaction strongly assoiciated with attrition (2.4 vs 2.7)
4. Employees with more previous companies are more likely leave again (2.9 vs 2.6)
5. Poor work life balance slightly associated with the attrition (2.7 vs 2.6) 
6. Single employees show highest attrition at 25.5%, nearly double that of married employees (12.5%).
7. Sales had highest attrition rate (20.6%) — not R&D as raw counts suggested
8. HR and Tech degrees had highest attrition rate(24-25%) - not life science as raw counts suggested
9. Younger employees leave more — average age 33 for quitters vs 37 for stayers.
10. Random Forest achieved 83% accuracy but only 6% Recall for quitters — proving accuracy is misleading on imbalanced datasets.

## Feature Engineering
Created two new features with business logic:
- **IncomePerYear** = MonthlyIncome / Age — captures compensation 
  relative to career stage. A 45-year-old earning 3000 is far more 
  frustrated than a 25-year-old earning the same.
- **SatisfactionScore** = average of JobSatisfaction, 
  EnvironmentSatisfaction, WorkLifeBalance — captures overall 
  dissatisfaction as a single metric.

Both features appeared in top 10 most important features 
confirmed by Random Forest feature importance.

## Model Comparison

| Model | Recall (quitters) | F1 | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.53 | 0.33 | 0.65 |
| Naive Bayes | 0.40 | 0.38 | 0.79 |
| SVM | 0.45 | 0.34 | 0.72 |
| Random Forest | 0.06 | 0.11 | 0.83 |
| LR Tuned (C=0.01) | 0.60 | 0.35 | 0.65 |

## Why Logistic Regression?
Recall was chosen as the primary metric because missing an 
employee who leaves is far more costly than a false alarm. 
A missed quitter costs recruitment and training expenses. 
A false alarm costs one HR conversation.

Despite Random Forest achieving 83% accuracy, its Recall of 
0.06 made it practically useless for this business problem — 
proving that accuracy is misleading on imbalanced datasets.

GridSearchCV revealed C=0.01 as optimal, achieving 0.68 Recall 
via 5-fold cross validation.

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (LogisticRegression, GaussianNB, SVC, RandomForest)
- Matplotlib, Seaborn
- StandardScaler, Pipeline, GridSearchCV

## Future Improvements
- Implement SMOTE for handling class imbalance
- Try XGBoost and LightGBM
- Deploy as Streamlit web application
- Add SHAP values for individual prediction explanations
