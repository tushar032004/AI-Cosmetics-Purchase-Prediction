# 📖 Data Dictionary: Cosmetics Customer Purchase Dataset

This document details the schema, definitions, measurement units, and categorical levels for the dataset used in the **AI-Based Cosmetic Purchase Prediction** project (`cleaned_cosmetics_dataset.csv`).

---

## 1. Feature Specifications

| Column Name | Data Type | Role | Description | Example Values |
| :--- | :--- | :--- | :--- | :--- |
| **`Id`** | Integer | Identifier | Unique customer identification number. Excluded from machine learning training. | `7363`, `2185`, `6157` |
| **`sex`** | String / Object | Feature (Demographic) | Biological sex / gender identity of the customer. | `Female`, `Male` |
| **`age_group`** | String / Object | Feature (Demographic) | Age bracket of the customer. Created by binning continuous ages into marketing categories. | `18-25`, `26-35`, `36-45`, `46-55`, `56+` |
| **`status`** | String / Object | Feature (Behavioral) | Current customer account status / relationship stage with the cosmetics brand. | `New`, `Active`, `Inactive` |
| **`tenure`** | Float | Feature (Loyalty) | Duration of time (in months) the customer has maintained an account or relationship with the brand. | `16.0`, `65.0`, `98.0` |
| **`total`** | Float | Feature (Financial) | Total cumulative historical spend (in currency units, e.g. USD / INR) across all past cosmetic purchases. | `260.0`, `770.0`, `3050.0` |
| **`region`** | String / Object | Feature (Geographic) | Geographic sales territory / regional market where the customer resides. | `North`, `South`, `East`, `West` |
| **`income`** | Float | Feature (Financial) | Estimated annual personal income of the customer. | `17600.0`, `49100.0`, `85400.0` |
| **`quantity`** | Integer | Feature (Behavioral) | Number of units purchased during previous cosmetic order transactions. | `1`, `3`, `6`, `8` |
| **`purchased`** | Integer | **Target Variable** | Binary outcome indicator representing whether the customer purchased the newly launched unisex skincare product (`1 = Yes`, `0 = No`). | `0`, `1` |

---

## 2. Age Group Interpretations

As prescribed in the project guide, continuous age values are categorized into five strategic cohorts:

* **`18–25`**: Young adult customers (early career, university students, high digital and social-media receptivity).
* **`26–35`**: Prime adult customers (established professionals, primary disposable skincare spenders).
* **`36–45`**: Mid-age adult customers (focus on skin wellness, maintenance, and barrier hydration).
* **`46–55`**: Older adult customers (preference for premium, targeted skincare solutions).
* **`56+`**: Senior customers (high brand loyalty, lower baseline propensity for unisex trends).

---

## 3. Data Cleaning & Preprocessing Notes

1. **Identifier Removal**: `Id` is dropped prior to model training to prevent artificial correlation.
2. **Categorical Encoding**: `sex`, `age_group`, `status`, and `region` are encoded via `OneHotEncoder(handle_unknown="ignore")`.
3. **Missing Value Handling**: Numeric fields are checked for missing values or extreme outliers during cleaning.
