# ============================================================
# AI COSMETIC PURCHASE PREDICTION
# STREAMLIT APPLICATION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cosmetics Purchase Prediction",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #faf9fc;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }
    h1 {
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    h2 {
        font-weight: 650;
        margin-top: 1.5rem;
    }
    h3 {
        font-weight: 600;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e9e5ef;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e9e5ef;
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 45px;
    }
    .highlight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .recommendation-box {
        background-color: #e6f4ea;
        border-left: 5px solid #34a853;
        padding: 18px 22px;
        border-radius: 8px;
        margin: 15px 0;
    }
    .ethics-box {
        background-color: #fef7e0;
        border-left: 5px solid #f9ab00;
        padding: 18px 22px;
        border-radius: 8px;
        margin: 15px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# PROJECT PATHS & DATA LOADING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "processed" / "cleaned_cosmetics_dataset.csv"

@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)

try:
    df = load_data()
except FileNotFoundError:
    st.error(f"Dataset not found at: {CSV_PATH}")
    st.stop()

# Required columns verification
required_columns = ["Id", "purchased", "sex", "age_group", "status", "region"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

# ============================================================
# MODEL TRAINING & CACHING
# ============================================================

X = df.drop(["Id", "purchased"], axis=1)
y = df["purchased"]
categorical_columns = ["sex", "age_group", "status", "region"]

@st.cache_resource
def train_models(data_hash):
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
        ],
        remainder="passthrough"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Decision Tree
    dt_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("decision_tree", DecisionTreeClassifier(max_depth=5, random_state=42))
    ])
    dt_pipeline.fit(X_train, y_train)

    # 2. Logistic Regression
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("logistic_regression", LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_pipeline.fit(X_train, y_train)

    # 3. Random Forest
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("random_forest", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"))
    ])
    rf_pipeline.fit(X_train, y_train)

    # Metrics computation helper
    def get_metrics(pipe):
        preds = pipe.predict(X_test)
        probas = pipe.predict_proba(X_test)[:, 1]
        return {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probas),
            "preds": preds,
            "probas": probas
        }

    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "dt": {"model": dt_pipeline, "metrics": get_metrics(dt_pipeline)},
        "lr": {"model": lr_pipeline, "metrics": get_metrics(lr_pipeline)},
        "rf": {"model": rf_pipeline, "metrics": get_metrics(rf_pipeline)}
    }

models_data = train_models(len(df))
X_train = models_data["X_train"]
X_test = models_data["X_test"]
y_train = models_data["y_train"]
y_test = models_data["y_test"]

dt_model, dt_metrics = models_data["dt"]["model"], models_data["dt"]["metrics"]
lr_model, lr_metrics = models_data["lr"]["model"], models_data["lr"]["metrics"]
rf_model, rf_metrics = models_data["rf"]["model"], models_data["rf"]["metrics"]

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("## 💄 Cosmetic AI\n**Unisex Skincare Purchase System**")
    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "🏠 Dashboard",
            "🎯 Business Recommendation",
            "🔮 Prediction",
            "🤖 Model Comparison",
            "🌳 Decision Tree",
            "🌲 Random Forest",
            "📈 Logistic Regression",
            "⚖️ Ethics & Unisex Insights",
            "📁 Project Files"
        ]
    )

    st.divider()
    st.caption("AI-Based Customer Purchase Prediction")
    st.caption("B.Tech CSE Machine Learning Project")

# ============================================================
# 🏠 DASHBOARD
# ============================================================

if page == "🏠 Dashboard":
    st.title("💄 AI Cosmetic Purchase Prediction")
    st.markdown(
        """
        ### Machine Learning & Business Intelligence Dashboard
        Predicting whether an individual customer will purchase a newly launched unisex skincare product, 
        and determining which age demographic holds the highest commercial potential.
        """
    )
    st.divider()

    total_customers = len(df)
    total_purchased = int(df["purchased"].sum())
    total_not_purchased = total_customers - total_purchased
    purchase_rate = (total_purchased / total_customers) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Customers", f"{total_customers:,}")
    col2.metric("🛍️ Purchased", f"{total_purchased:,}")
    col3.metric("❌ Not Purchased", f"{total_not_purchased:,}")
    col4.metric("📊 Conversion Rate", f"{purchase_rate:.2f}%")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📊 Purchase Distribution")
        purchase_counts = df["purchased"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(["Not Purchased", "Purchased"], [purchase_counts.get(0, 0), purchase_counts.get(1, 0)], color=["#ff9999", "#66b3ff"])
        ax.set_ylabel("Customer Count")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 50, f"{int(h):,}", ha="center", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.subheader("👥 Historical Purchases by Age Group")
        age_purchase = pd.crosstab(df["age_group"], df["purchased"])
        fig, ax = plt.subplots(figsize=(7, 4))
        age_purchase.plot(kind="bar", stacked=True, ax=ax, color=["#e0e0e0", "#764ba2"])
        ax.set_ylabel("Customers")
        ax.legend(["Not Purchased", "Purchased"])
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ============================================================
# 🎯 BUSINESS RECOMMENDATION (CORE OBJECTIVE)
# ============================================================

elif page == "🎯 Business Recommendation":
    st.title("🎯 Age Group Buyer Potential & Business Recommendation")
    st.markdown(
        """
        According to the core business objective: **AI predicts customer purchase probability; 
        business analysis aggregates those probabilities to determine the age category with the highest expected number of buyers.**
        """
    )
    st.divider()

    # Model selector for aggregation
    chosen_model_name = st.selectbox(
        "Select Model for Probability Aggregation:",
        ["Random Forest (Recommended)", "Logistic Regression", "Decision Tree"]
    )
    active_pipe = rf_model if "Random" in chosen_model_name else (lr_model if "Logistic" in chosen_model_name else dt_model)

    # Calculate probabilities
    full_probas = active_pipe.predict_proba(X)[:, 1]
    df_eval = df.copy()
    df_eval["purchase_probability"] = full_probas

    # Group by age_group
    age_agg = (
        df_eval.groupby("age_group", observed=False)["purchase_probability"]
        .agg(["count", "sum", "mean"])
        .rename(columns={
            "count": "Total Customers",
            "sum": "Expected Buyers",
            "mean": "Avg Probability"
        })
    )
    total_expected = age_agg["Expected Buyers"].sum()
    age_agg["Potential Buyers Share (%)"] = ((age_agg["Expected Buyers"] / total_expected) * 100).round(2)
    age_agg["Avg Probability (%)"] = (age_agg["Avg Probability"] * 100).round(2)
    age_agg["Expected Buyers"] = age_agg["Expected Buyers"].round(1)
    age_agg = age_agg.sort_values(by="Potential Buyers Share (%)", ascending=False)

    top_group = age_agg.index[0]
    top_share = age_agg.iloc[0]["Potential Buyers Share (%)"]
    top_expected = age_agg.iloc[0]["Expected Buyers"]

    # Highlights
    st.markdown(
        f"""
        <div class="recommendation-box">
            <h3 style="margin:0; color:#137333;">🏆 Primary Marketing Target: {top_group} Age Category</h3>
            <p style="margin:5px 0 0 0; font-size:16px;">
                The <strong>{top_group}</strong> age category represents <strong>{top_share}%</strong> of all predicted buyers 
                (approximately <strong>{top_expected:,.0f}</strong> expected customers). 
                The AI team recommends concentrating the new unisex product launch marketing budget on this demographic segment.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("📋 Expected Buyer Distribution Table (Guide Sections 7 & 8)")
    st.dataframe(
        age_agg[["Total Customers", "Expected Buyers", "Avg Probability (%)", "Potential Buyers Share (%)"]],
        use_container_width=True
    )

    # Chart
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(age_agg.index, age_agg["Potential Buyers Share (%)"], color="#667eea")
    ax.set_title(f"Estimated Share of Potential Buyers by Age Group ({chosen_model_name})", fontsize=14, fontweight="bold")
    ax.set_ylabel("Share of Potential Buyers (%)")
    ax.set_xlabel("Age Group")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha="center", fontweight="bold")
    ax.set_ylim(0, max(age_agg["Potential Buyers Share (%)"]) + 10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # Decision Threshold Simulation
    st.subheader("⚙️ Decision Threshold Sensitivity Simulator (Investigative Question)")
    st.markdown(
        "In direct marketing, a 0.5 threshold is often suboptimal because reaching out to a potential buyer is cheap, but missing one is expensive. Move the slider to see how targeted customer volume changes:"
    )

    threshold = st.slider("Classification Decision Threshold", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
    targeted_mask = df_eval["purchase_probability"] >= threshold
    targeted_count = targeted_mask.sum()
    targeted_pct = (targeted_count / len(df_eval)) * 100

    col_t1, col_t2 = st.columns(2)
    col_t1.metric("🎯 Customers Flagged for Outreach", f"{targeted_count:,}", f"{targeted_pct:.1f}% of total")
    col_t2.metric("Targeting Threshold", f"{threshold:.2f}")

# ============================================================
# 🔮 PREDICTION
# ============================================================

elif page == "🔮 Prediction":
    st.title("🔮 Customer Purchase Prediction")
    st.markdown("Enter individual customer details to predict purchase likelihood.")
    st.divider()

    model_choice = st.selectbox(
        "Choose Machine Learning Model",
        ["Random Forest", "Logistic Regression", "Decision Tree"]
    )
    selected_model = rf_model if model_choice == "Random Forest" else (lr_model if model_choice == "Logistic Regression" else dt_model)

    input_values = {}
    input_cols = st.columns(2)
    feature_cols = [col for col in X.columns]

    for idx, col in enumerate(feature_cols):
        with input_cols[idx % 2]:
            if col in categorical_columns:
                options = sorted(X[col].dropna().unique().tolist())
                input_values[col] = st.selectbox(col.replace("_", " ").title(), options)
            else:
                input_values[col] = st.number_input(
                    col.replace("_", " ").title(),
                    value=float(X[col].median())
                )

    if st.button("🔮 Predict Purchase Propensity", use_container_width=True):
        input_df = pd.DataFrame([input_values])[X.columns]
        pred = selected_model.predict(input_df)[0]
        prob = selected_model.predict_proba(input_df)[0][1]

        st.divider()
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            if pred == 1:
                st.success("🛍️ **High Propensity**: Customer is likely to purchase!")
            else:
                st.info("ℹ️ **Low Propensity**: Customer is unlikely to purchase.")
        with res_col2:
            st.metric("Purchase Probability", f"{prob * 100:.2f}%")

        st.progress(float(prob))

# ============================================================
# 🤖 MODEL COMPARISON
# ============================================================

elif page == "🤖 Model Comparison":
    st.title("🤖 Machine Learning Model Comparison")
    st.markdown("Comprehensive evaluation across all three classification algorithms:")
    st.divider()

    results_df = pd.DataFrame({
        "Model": ["Decision Tree", "Logistic Regression", "Random Forest"],
        "Accuracy (%)": [dt_metrics["accuracy"]*100, lr_metrics["accuracy"]*100, rf_metrics["accuracy"]*100],
        "Precision (%)": [dt_metrics["precision"]*100, lr_metrics["precision"]*100, rf_metrics["precision"]*100],
        "Recall (%)": [dt_metrics["recall"]*100, lr_metrics["recall"]*100, rf_metrics["recall"]*100],
        "F1-Score (%)": [dt_metrics["f1"]*100, lr_metrics["f1"]*100, rf_metrics["f1"]*100],
        "ROC-AUC": [dt_metrics["roc_auc"], lr_metrics["roc_auc"], rf_metrics["roc_auc"]]
    }).round(2)

    st.dataframe(results_df, use_container_width=True, hide_index=True)

    # Bar chart of metrics
    fig, ax = plt.subplots(figsize=(11, 5))
    metrics_names = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]
    x = np.arange(len(metrics_names))
    width = 0.25

    ax.bar(x - width, results_df.iloc[0][metrics_names], width, label="Decision Tree", color="#66b3ff")
    ax.bar(x, results_df.iloc[1][metrics_names], width, label="Logistic Regression", color="#ff9999")
    ax.bar(x + width, results_df.iloc[2][metrics_names], width, label="Random Forest", color="#2ca02c")

    ax.set_ylabel("Score (%)")
    ax.set_title("Model Evaluation Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# 🌳 DECISION TREE
# ============================================================

elif page == "🌳 Decision Tree":
    st.title("🌳 Decision Tree Model")
    st.markdown("Interpretable tree-based classification.")
    st.divider()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{dt_metrics['accuracy']*100:.2f}%")
    m2.metric("Precision", f"{dt_metrics['precision']*100:.2f}%")
    m3.metric("Recall", f"{dt_metrics['recall']*100:.2f}%")
    m4.metric("F1-Score", f"{dt_metrics['f1']*100:.2f}%")
    m5.metric("ROC-AUC", f"{dt_metrics['roc_auc']:.4f}")

    st.divider()
    st.subheader("🌳 Decision Tree Structure (First 3 Levels)")
    trained_tree = dt_model.named_steps["decision_tree"]
    feat_names = dt_model.named_steps["preprocessor"].get_feature_names_out()

    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(
        trained_tree,
        feature_names=feat_names,
        class_names=["Not Purchased", "Purchased"],
        filled=True,
        max_depth=3,
        fontsize=11,
        rounded=True,
        ax=ax
    )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# 🌲 RANDOM FOREST
# ============================================================

elif page == "🌲 Random Forest":
    st.title("🌲 Random Forest Classifier")
    st.markdown("Ensemble learning model providing robust predictive power and feature importance.")
    st.divider()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{rf_metrics['accuracy']*100:.2f}%")
    m2.metric("Precision", f"{rf_metrics['precision']*100:.2f}%")
    m3.metric("Recall", f"{rf_metrics['recall']*100:.2f}%")
    m4.metric("F1-Score", f"{rf_metrics['f1']*100:.2f}%")
    m5.metric("ROC-AUC", f"{rf_metrics['roc_auc']:.4f}")

    st.divider()
    st.subheader("📌 Random Forest Feature Importances")
    feat_names = rf_model.named_steps["preprocessor"].get_feature_names_out()
    importances = rf_model.named_steps["random_forest"].feature_importances_

    imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values(by="Importance", ascending=False)

    col_f1, col_f2 = st.columns([1, 1.5])
    with col_f1:
        st.dataframe(imp_df.head(12).reset_index(drop=True), use_container_width=True)

    with col_f2:
        fig, ax = plt.subplots(figsize=(8, 6))
        top_10 = imp_df.head(10).sort_values(by="Importance", ascending=True)
        ax.barh(top_10["Feature"], top_10["Importance"], color="#2ca02c")
        ax.set_xlabel("Importance")
        ax.set_title("Top 10 Feature Importances", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ============================================================
# 📈 LOGISTIC REGRESSION
# ============================================================

elif page == "📈 Logistic Regression":
    st.title("📈 Logistic Regression Model")
    st.markdown("Probabilistic linear baseline for binary purchase classification.")
    st.divider()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{lr_metrics['accuracy']*100:.2f}%")
    m2.metric("Precision", f"{lr_metrics['precision']*100:.2f}%")
    m3.metric("Recall", f"{lr_metrics['recall']*100:.2f}%")
    m4.metric("F1-Score", f"{lr_metrics['f1']*100:.2f}%")
    m5.metric("ROC-AUC", f"{lr_metrics['roc_auc']:.4f}")

    st.divider()
    st.subheader("🔢 Confusion Matrix")
    cm = confusion_matrix(y_test, lr_metrics["preds"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Purples")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Purchased", "Purchased"])
    ax.set_yticklabels(["Not Purchased", "Purchased"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ============================================================
# ⚖️ ETHICS & UNISEX INSIGHTS (GUIDE SECTION 10 & 14)
# ============================================================

elif page == "⚖️ Ethics & Unisex Insights":
    st.title("⚖️ Ethics, Responsible AI & Unisex Product Insights")
    st.markdown(
        """
        ### Responsible Machine Learning in Cosmetics & Skincare
        As highlighted in the Student Teaching Guide (Section 10), demographic-based machine learning requires strong ethical governance.
        """
    )
    st.divider()

    st.markdown(
        """
        <div class="ethics-box">
            <h4 style="margin:0; color:#b06000;">⚠️ Purchase Propensity vs. Stereotyping</h4>
            <p style="margin:6px 0 0 0;">
                This project models <strong>purchase propensity</strong> (likelihood of commercial transaction based on historical engagement), 
                <strong>NOT</strong> a value judgment that any demographic group, age bracket, or gender "needs" appearance alteration.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("1. Unisex Skincare Market Dynamics")
    st.write(
        """
        * **Breaking Stigmas**: Historically, skincare marketing disproportionately targeted women. Unisex formulations create neutral, health-first positioning.
        * **Cross-Demographic Adoption**: Our models observe that younger male cohorts (18–25, 26–35) demonstrate significantly higher purchase propensity for unisex skincare compared to older cohorts (56+).
        * **Inclusive Messaging**: Marketing should emphasize dermatological health and hydration rather than anti-aging panic or gender-conforming beauty standards.
        """
    )

    st.subheader("2. Ethical Data Governance & Privacy Considerations")
    st.write(
        """
        * **Consent & Transparency**: Customer data should only be gathered with explicit opt-in for product recommendation and analytics.
        * **Avoid Algorithmic Redlining**: Ensuring models do not artificially exclude entire communities or regions from receiving product promotions.
        * **Representativeness**: Training models on synthetic or unrepresentative data risks biased marketing assumptions; real-world deployment requires continuous fairness monitoring.
        """
    )

# ============================================================
# 📁 PROJECT FILES
# ============================================================

elif page == "📁 Project Files":
    st.title("📁 Project Files")
    st.markdown("Browse and download codebase files and documentation.")
    st.divider()

    folders = ["data", "models", "notebooks", "reports", "docs"]
    for folder_name in folders:
        folder_path = BASE_DIR / folder_name
        st.subheader(f"📂 {folder_name}")
        if not folder_path.exists():
            continue
        files = [f for f in folder_path.rglob("*") if f.is_file()]
        if not files:
            st.info("No files found.")
            continue
        for file_path in files:
            rel_path = file_path.relative_to(BASE_DIR)
            with st.expander(f"📄 {rel_path}"):
                ext = file_path.suffix.lower()
                if ext == ".py":
                    st.code(file_path.read_text(encoding="utf-8", errors="replace"), language="python")
                elif ext == ".csv":
                    st.dataframe(pd.read_csv(file_path).head(50))
                elif ext in [".md", ".txt"]:
                    st.markdown(file_path.read_text(encoding="utf-8", errors="replace"))
                try:
                    with open(file_path, "rb") as f:
                        st.download_button(f"⬇️ Download {file_path.name}", f.read(), file_name=file_path.name, key=f"dl_{rel_path}")
                except Exception:
                    pass

st.divider()
st.caption("💄 AI Cosmetic Purchase Prediction | Decision Tree + Random Forest + Logistic Regression")