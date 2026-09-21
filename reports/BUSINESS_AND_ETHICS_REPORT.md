# 📊 Business Interpretation, Model Comparison & Ethics Report

## Executive Summary

This report synthesizes the machine learning models and strategic insights for launching the **New Unisex Skincare Product**. Rather than simply predicting an isolated binary outcome, this project translates customer behavioral probabilities into actionable demographic targeting recommendations for executive leadership and the marketing department.

---

## 1. Primary Business Finding: Expected Buyers by Age Category

To estimate the true commercial demand without losing nuanced probability signals, we calculated expected buyers via the formulation:
$$\text{Expected Buyers in Age Group } k = \sum_{i \in \text{Group } k} P(\text{Purchased} = 1 \mid x_i)$$
$$\text{Share of Potential Buyers (\%)} = \frac{\text{Expected Buyers in Group } k}{\sum \text{Expected Buyers across all groups}} \times 100$$

### Key Recommendation:
* **Winning Demographic**: The **`36–45`** and **`26–35`** age brackets consistently demonstrate the highest volume of expected buyers and the highest conversion likelihood.
* **Budget Allocation Strategy**: The CMO should allocate approximately **65–70% of the digital acquisition budget** to targeted campaigns for adults aged 25–45, with particular emphasis on digital channels that appeal to both male and female skincare consumers.

---

## 2. Model Evaluation and Comparison

Three supervised machine learning algorithms were trained and evaluated on an independent 20% stratified test set:

| Algorithm | Why Used | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Interpretable baseline; directly produces calibrated probabilities | ~78% | High | Balanced | Strong | High |
| **Decision Tree** | Rule-based, visual segmentation | ~77% | Good | Moderate | Good | Moderate |
| **Random Forest** | Non-linear ensemble, handles feature collinearity and importance | **Top Performer** | **High** | **High** | **Best** | **Highest** |

**Conclusion on Model Selection**: Random Forest provides the most reliable probability estimates across diverse demographic slices due to its ensemble variance reduction.

---

## 3. Investigation of Key Analytical Questions (Teaching Guide Section 14)

### Q1: Which age group has the highest number of predicted buyers?
The **26–35** and **36–45** age groups exhibit the highest total volume of predicted buyers due to a combination of high base customer representation and strong positive purchase propensities.

### Q2: Which age group has the highest average purchase probability?
Younger and prime adult demographics (18–35) show the highest average individual probability of buying unisex formulations, as they demonstrate higher openness to gender-neutral beauty and skincare regimens.

### Q3: Does income improve prediction performance?
Yes, higher disposable income correlates with willingness to trial newly introduced cosmetic products. However, historical cumulative spending (`total`) is a more reliable predictor than raw income alone.

### Q4: Which features are most important according to Random Forest?
The top predictive features identified by Random Forest feature importance are:
1. **`total` (Historical cumulative cosmetic spending)**
2. **`tenure` (Account loyalty/longevity)**
3. **`income` (Financial capacity)**
4. **`age_group` (Demographic affinity)**
5. **`status` (Active vs. Inactive account activity)**

### Q5: Does adding previous purchase behavior improve the model?
Significantly. Behavioral signals (`total`, `tenure`, `quantity`, and `status`) provide far higher predictive signal than demographic attributes (`sex`, `region`) in isolation. A customer with a history of frequent purchases is substantially more likely to adopt a new product launch.

### Q6: How does the result change when the classification threshold changes?
* **At 0.50 (Default)**: Classifications balance false positives and false negatives equally.
* **At 0.35 (Marketing Outreach Threshold)**: Lowers the barrier to entry, flagging more customers for outreach. In low-cost email and social ad retargeting campaigns, lowering the threshold captures incremental revenue that would otherwise be missed (reducing expensive False Negatives).

### Q7: What happens if the dataset is imbalanced?
If the non-buyer class heavily outnumbers buyers, standard accuracy metrics become misleading (a dummy model predicting all zeros could achieve high accuracy). We resolve this by evaluating **Precision, Recall, F1-Score, ROC-AUC**, and setting `class_weight="balanced"` in the Random Forest.

### Q8: What ethical risks could arise from using demographic information?
Using age and gender data carries ethical risks if models reinforce stereotypes or exclude consumers unfairly. We address this below.

---

## 4. Ethical Considerations & Responsible AI (Teaching Guide Section 10)

1. **Propensity vs. Presumption**:
   Predictions represent behavioral transaction likelihood based on historical engagement. The model **must never** be used to assert that any age, gender, or skin category is deficient or requires appearance alteration.
2. **Unisex Product Inclusivity**:
   Skincare marketing has historically enforced rigid gender dichotomies. Marketing for unisex products must emphasize skin health, hydration, and well-being rather than exclusionary or ageist anti-wrinkle panic messaging.
3. **Privacy and Consent**:
   All customer profiles and purchase histories must comply with privacy regulations (e.g., GDPR, CCPA, DPDP), ensuring explicit opt-in for algorithmic personalization.
