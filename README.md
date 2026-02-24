## LSEG Case Study – Attrition Analysis

**Senior People Data Scientist Submission**

---

## Project Objective

The objective of this case study is to analyze workforce data from Westworld Group to:

* Understand workforce composition
* Identify structural drivers of attrition
* Distinguish controllable vs external attrition factors
* Develop a predictive model for attrition risk
* Provide strategic recommendations for HR leadership

The analysis integrates quantitative modeling and qualitative exit survey insights.

---

## Dataset Description

The analysis uses two datasets:

1. **HRIS Extract**

   * Employee demographics
   * Job attributes
   * Compensation
   * Satisfaction metrics
   * Attrition indicator

2. **Exit Survey Data**

   * Employee ID
   * Exit statement (qualitative text)

The HRIS dataset contains ~1,470 employee records.

---

## Project Structure

LSEG_CASE_STUDY/
│
├── data/
│   ├── raw/
│   │   ├── exit_survey.csv
│   │   ├── hris_extract.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Modeling.ipynb
│   ├── 03_Exit_Survey_Analysis.ipynb
│
│
├── output/
│   ├── eda_figures/
│   │   ├── attrition_by_business_travel.png 
│   │   ├── attrition_by_job_level.png
│   │   ├── attrition_by_overtime.png
│   │   ├── attrition_by_overtime.png 
│   │   ├── controllable_exit_survey.png 
│   │   └── External_vs_Controlled.png
│   │
│
├── presentation/
│   ├── Workforce_Insights_Presentation.pptx
│
├── scripts
│   ├── convert_excel_to_csv.py
│
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── modeling.py
│
├── requirements.txt
└── README.md

---

## Methodology

### Step 1 – Data Cleaning

* Removed constant columns
* Handled missing values
* Converted categorical features
* Ensured proper data types

### Step 2 – Exploratory Data Analysis

* Workforce distribution analysis
* Attrition segmentation (Department, Job Level, Overtime, Travel)
* Satisfaction and compensation patterns

### Step 3 – Predictive Modeling

Models evaluated:

* Logistic Regression
* L1-Regularized Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

Logistic Regression achieved:

* ROC-AUC ≈ 0.80
* Strong interpretability

Tree-based models were evaluated for nonlinear effects but did not materially outperform logistic regression.

### Step 4 – Exit Survey Analysis

* Text preprocessing
* Theme classification
* Separation of:

  * External / Personal exits
  * Controllable organizational drivers

---

## Key Findings

* Attrition is structurally concentrated among:

  * Junior employees
  * Sales roles
  * Employees working overtime
  * Frequent travelers

* 36% of exits are driven by external or personal reasons.

* Among controllable factors, career growth is the dominant driver.

* Compensation is not the primary perceived reason for attrition.

* Satisfaction metrics act as protective factors.

Attrition is predictable and partially controllable.

---

## Strategic Recommendations

1. Strengthen career mobility pathways
2. Reduce workload and travel intensity
3. Enhance early-tenure engagement
4. Implement predictive retention monitoring

---

## How to Run the Project

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Notebooks

Open Jupyter Notebook:

```bash
jupyter notebook
```

Execute notebooks in order:

1. 01_EDA.ipynb
2. 02_Modeling.ipynb
3. 03_Exit_Survey_Analysis.ipynb

---

## Notes

* Hyperparameters were selected conservatively to prevent overfitting given dataset size.
* Further cross-validated hyperparameter tuning can be implemented in production settings.
* The predictive model is designed for interpretability and business applicability.

---

