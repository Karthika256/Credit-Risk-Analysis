# Credit-Risk-Analysis
A data science project on predicting loan defaults using logistic regression, Random Forest and XGBoost.
# Introduction

# Project Title: Credit Risk Analysis
## Objective
To predict loan default likelihood based on past financial and demographic data, helping financial institutions mitigate risks effectively.
## Background
Credit risk is a major concern for financial institutions. Identifying key factors that influence loan default can lead to better risk assessment and lending policies.

# Dataset Overview
The dataset consists of 887,379 records and 74 variables:
* 51 Numeric Variables
* 23 Categorical Variables
Some variables have a high percentage of missing values, requiring further analysis for imputation or removal.

# Exploratory Data Analysis
## Understanding the Target Variable
Target Variable: loan_status
- Current and Issued statuses provide inconclusive outcomes, so these records are removed.
- Rows with NA values in loan_status are also discarded.
- Final dataset: 277,140 records
- New Binary Variable Created:
loan_status_binary:
1 (Fully Paid) – Good loan
0 (Others) – Bad loan
## Feature Selection
### Discarding Redundant Variables
Some variables provide no predictive value and are removed
### Handling Missing Data
- Strategy Based on Missingness Level
Missingness Level
*0-20% (Low) - Impute values
*50-80% (High) - Create a new binary missingness variable
*95-99% (Very High) - Discard variable
### Handling Date Variables
* Converted all date variables into duration (days/months) from the most recent date.
### Outliers & Skewness Handling
- Applied Winsorization and transformations:
- Log transformation for skewness > 10
- Square root transformation for skewness > 1
- Converted variables with >80% zero values into binary categorical variables
### Encoding Categorical Variables
- One-Hot Encoding: term, home_ownership, verification_status, pymnt_plan, purpose, addr_state, initial_list_status, application_type
- Ordinal Encoding: grade, emp_length
- Dropped subgrade due to collinearity with grade
### Feature Correlation & Selection
- Highly Correlated Variables Removed
### Addressing Class Imbalance
Split dataset into training and test set
SMOTE (Synthetic Minority Over-sampling Technique) applied to the training set to balance classes.

# Model Selection & Performance Evaluation
Model Comparison
1. AUC - ROC
* Logistic Regression - 0.978
* Random Forest - 0.997
* XGBoost - 0.999
3. Sensitivity (Recall)
* Logistic Regression - 0.978
* Random Forest - 0.950
* XGBoost - 0.983
Best Model: XGBoost – Highest AUC-ROC with competitive recall.

# Key Features & Interpretation
1. **total_pymnt**
* Coefficient : +10.16
* Higher payments indicate a higher likelihood of default (unexpected finding)
  
2. **int_rate**
* Coefficient : +1.29
* Higher interest rates increase default risk
  
3. **loan_amnt**
* Coefficient : -4.57
* Larger loans decrease default risk (strong borrowers)
  
4. **dti**
* Coefficient : -0.07
* Higher DTI surprisingly reduces default risk
  
5. **emp_length**
* Coefficient : -0.03
* Longer employment reduces default risk

# Real-World Implications
1. **Interest Rates & Risk** – Higher interest rates strongly predict defaults. Lenders may need to adjust interest rates to lower-risk borrowers.
2. **Loan Amounts** – Larger loans tend to be safer; banks can target high-income borrowers for larger loans.
3. **Loan Purpose Matters** – Vacations & personal loans show higher risk; home improvement and medical loans are safer.
4. **State-Level Risk** – Some states show higher default rates, requiring adjusted lending policies.
5. **Debt-to-Income Ratio** – Traditional assumptions about DTI and default risk may need reconsideration.

# Conclusion
This analysis offers real-world insights into loan default risk, enabling lenders to refine credit scoring models, adjust eligibility criteria, and develop targeted lending strategies. By identifying high-risk borrowers early, financial institutions can minimize losses and optimize loan approval processes. Future steps include testing more advanced machine learning models, implementing risk-based pricing strategies, and exploring external economic factors influencing default rates.

