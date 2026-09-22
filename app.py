# ============================================================
# AI COSMETIC PURCHASE PREDICTION
# STREAMLIT APPLICATION  –  v3.0 (Dark Mode + Professional UI)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Cosmetic Purchase Prediction",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DARK MODE TOGGLE  (session state)
# ============================================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# ============================================================
# THEME VARIABLES
# ============================================================

if st.session_state.dark_mode:
    BG          = "#0f1117"
    CARD_BG     = "#1e2130"
    SIDEBAR_BG  = "#161b2e"
    TEXT        = "#e8eaf6"
    SUB_TEXT    = "#9aa3b5"
    BORDER      = "#2d3354"
    ACCENT      = "#7c6af7"
    ACCENT2     = "#f472b6"
    SUCCESS_BG  = "#0d2a1e"
    SUCCESS_BD  = "#22c55e"
    WARN_BG     = "#2a1f0d"
    WARN_BD     = "#f59e0b"
    MPL_STYLE   = "dark_background"
    PLOT_BG     = "#1e2130"
    PLOT_FG     = "#e8eaf6"
else:
    BG          = "#f7f8fc"
    CARD_BG     = "#ffffff"
    SIDEBAR_BG  = "#ffffff"
    TEXT        = "#1e293b"
    SUB_TEXT    = "#64748b"
    BORDER      = "#e2e8f0"
    ACCENT      = "#6366f1"
    ACCENT2     = "#ec4899"
    SUCCESS_BG  = "#f0fdf4"
    SUCCESS_BD  = "#22c55e"
    WARN_BG     = "#fffbeb"
    WARN_BD     = "#f59e0b"
    MPL_STYLE   = "seaborn-v0_8-whitegrid"
    PLOT_BG     = "#ffffff"
    PLOT_FG     = "#1e293b"

# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(f"""
<style>
  /* ---- Page background ---- */
  .stApp {{ background-color: {BG}; color: {TEXT}; }}
  .block-container {{ padding: 2rem 2.5rem 3rem; }}

  /* ---- Sidebar ---- */
  section[data-testid="stSidebar"] {{
      background-color: {SIDEBAR_BG};
      border-right: 1px solid {BORDER};
  }}
  section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

  /* ---- Metric cards ---- */
  div[data-testid="metric-container"] {{
      background-color: {CARD_BG};
      border: 1px solid {BORDER};
      padding: 16px 20px;
      border-radius: 14px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.08);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
  }}
  div[data-testid="metric-container"]:hover {{
      transform: translateY(-3px);
      box-shadow: 0 8px 20px rgba(0,0,0,0.12);
  }}
  div[data-testid="metric-container"] label {{ color: {SUB_TEXT} !important; font-size: 13px !important; }}
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {{ color: {TEXT} !important; font-weight: 700; }}

  /* ---- Buttons ---- */
  .stButton > button {{
      border-radius: 10px;
      font-weight: 600;
      min-height: 46px;
      background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT2} 100%);
      color: white !important;
      border: none;
      transition: opacity 0.2s ease, transform 0.15s ease;
  }}
  .stButton > button:hover {{
      opacity: 0.88;
      transform: translateY(-2px);
  }}

  /* ---- DataFrames ---- */
  .stDataFrame {{ border-radius: 12px; overflow: hidden; }}

  /* ---- Headings ---- */
  h1 {{ font-weight: 800; letter-spacing: -0.5px; color: {TEXT}; }}
  h2 {{ font-weight: 700; color: {TEXT}; margin-top: 1.5rem; }}
  h3 {{ font-weight: 600; color: {TEXT}; }}
  p, li, label {{ color: {TEXT}; }}

  /* ---- Custom card classes ---- */
  .hero-card {{
      background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT2} 100%);
      color: white;
      padding: 28px 32px;
      border-radius: 16px;
      margin-bottom: 24px;
      animation: slideDown 0.5s ease;
  }}
  .info-card {{
      background-color: {CARD_BG};
      border: 1px solid {BORDER};
      border-left: 5px solid {ACCENT};
      padding: 20px 24px;
      border-radius: 10px;
      margin: 14px 0;
      animation: fadeIn 0.4s ease;
  }}
  .success-card {{
      background-color: {SUCCESS_BG};
      border-left: 5px solid {SUCCESS_BD};
      padding: 18px 22px;
      border-radius: 10px;
      margin: 14px 0;
      animation: fadeIn 0.4s ease;
  }}
  .warning-card {{
      background-color: {WARN_BG};
      border-left: 5px solid {WARN_BD};
      padding: 18px 22px;
      border-radius: 10px;
      margin: 14px 0;
  }}
  .stat-pill {{
      display: inline-block;
      background: linear-gradient(90deg, {ACCENT}33, {ACCENT2}33);
      border: 1px solid {ACCENT}66;
      padding: 4px 14px;
      border-radius: 20px;
      font-size: 13px;
      font-weight: 600;
      color: {ACCENT};
      margin: 3px 4px;
  }}

  /* ---- Animations ---- */
  @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(10px); }}
      to   {{ opacity: 1; transform: translateY(0); }}
  }}
  @keyframes slideDown {{
      from {{ opacity: 0; transform: translateY(-16px); }}
      to   {{ opacity: 1; transform: translateY(0); }}
  }}
  @keyframes pulse {{
      0%,100% {{ opacity: 1; }}
      50%      {{ opacity: 0.7; }}
  }}
  .page-content {{
      animation: fadeIn 0.45s ease;
  }}

  /* ---- Selectbox / inputs ---- */
  .stSelectbox > div, .stNumberInput > div {{
      border-radius: 8px;
  }}

  /* ---- Divider ---- */
  hr {{ border-color: {BORDER}; }}

  /* ---- Caption ---- */
  .stCaption {{ color: {SUB_TEXT}; }}

  /* ---- Progress bar ---- */
  .stProgress > div > div {{ background: linear-gradient(90deg, {ACCENT}, {ACCENT2}); border-radius: 10px; }}

  /* ---- Tabs ---- */
  .stTabs [data-baseweb="tab"] {{
      border-radius: 8px 8px 0 0;
      font-weight: 600;
      color: {SUB_TEXT};
  }}
  .stTabs [aria-selected="true"] {{
      background-color: {CARD_BG};
      color: {ACCENT} !important;
      border-bottom: 3px solid {ACCENT};
  }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS & DATA LOADING
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

required_columns = ["Id", "purchased", "sex", "age_group", "status", "region"]
missing_cols = [c for c in required_columns if c not in df.columns]
if missing_cols:
    st.error(f"Missing columns: {missing_cols}")
    st.stop()

# ============================================================
# MODEL TRAINING & CACHING
# ============================================================

X_all = df.drop(["Id", "purchased"], axis=1)
y_all = df["purchased"]
CATEGORICAL_COLS = ["sex", "age_group", "status", "region"]

@st.cache_resource
def train_models(data_hash):
    # ⚠️ IMPORTANT: Each pipeline MUST get its OWN preprocessor instance.
    # Sharing one ColumnTransformer object means the last .fit() call overwrites
    # the fitted state for all three pipelines — causing incorrect feature names
    # and transform outputs. We use a factory function to prevent this.
    def make_preprocessor():
        return ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS)
            ],
            remainder="passthrough"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # --- Decision Tree ---
    dt = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("decision_tree", DecisionTreeClassifier(max_depth=5, random_state=42))
    ])
    dt.fit(X_train, y_train)

    # --- Logistic Regression ---
    lr = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("logistic_regression", LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr.fit(X_train, y_train)

    # --- Random Forest (GridSearch optimised: 320 candidates, 5-fold CV) ---
    # Best params: n_estimators=300, max_depth=8, min_samples_split=5,
    #              min_samples_leaf=2, max_features='sqrt', class_weight=None
    rf = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("random_forest", RandomForestClassifier(
            n_estimators=300,       # 300 trees — best from GridSearch
            max_depth=8,            # prevents overfitting on this dataset
            min_samples_split=5,    # min samples needed to split a node
            min_samples_leaf=2,     # min samples required in each leaf
            max_features="sqrt",    # √features per split (standard RF)
            class_weight=None,      # no class weighting → maximises accuracy
            random_state=42
        ))
    ])
    rf.fit(X_train, y_train)

    def metrics(pipe):
        preds  = pipe.predict(X_test)
        probas = pipe.predict_proba(X_test)[:, 1]
        return {
            "accuracy" : accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall"   : recall_score(y_test, preds, zero_division=0),
            "f1"       : f1_score(y_test, preds, zero_division=0),
            "roc_auc"  : roc_auc_score(y_test, probas),
            "preds"    : preds,
            "probas"   : probas,
        }

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "dt": {"model": dt, "metrics": metrics(dt)},
        "lr": {"model": lr, "metrics": metrics(lr)},
        "rf": {"model": rf, "metrics": metrics(rf)},
    }

models_data = train_models(len(df))
X_train = models_data["X_train"];  X_test  = models_data["X_test"]
y_train = models_data["y_train"];  y_test  = models_data["y_test"]

dt_model, dt_m = models_data["dt"]["model"], models_data["dt"]["metrics"]
lr_model, lr_m = models_data["lr"]["model"], models_data["lr"]["metrics"]
rf_model, rf_m = models_data["rf"]["model"], models_data["rf"]["metrics"]

# ============================================================
# HELPER: matplotlib figure style matching theme
# ============================================================

def styled_fig(*args, **kwargs):
    """Return a plt.subplots with theme-aware facecolors."""
    matplotlib.rcParams.update({
        "figure.facecolor" : PLOT_BG,
        "axes.facecolor"   : PLOT_BG,
        "axes.edgecolor"   : BORDER,
        "text.color"       : PLOT_FG,
        "xtick.color"      : PLOT_FG,
        "ytick.color"      : PLOT_FG,
        "axes.labelcolor"  : PLOT_FG,
        "grid.color"       : BORDER,
        "axes.titlecolor"  : PLOT_FG,
    })
    return plt.subplots(*args, **kwargs)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    # Dark mode toggle
    dm_label = "☀️ Light Mode" if st.session_state.dark_mode else "🌙 Dark Mode"
    if st.button(dm_label, use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown(f"""
    <div style="text-align:center; padding:16px 0 8px;">
      <div style="font-size:36px;">💄</div>
      <div style="font-weight:800; font-size:17px; color:{ACCENT};">Cosmetic AI</div>
      <div style="font-size:12px; color:{SUB_TEXT}; margin-top:4px;">Unisex Skincare Purchase System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<hr style='border-color:{BORDER}; margin:8px 0;'>", unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🔍 EDA Explorer",
            "🎯 Business Recommendation",
            "🔮 Prediction",
            "🤖 Model Comparison",
            "🌳 Decision Tree",
            "🌲 Random Forest",
            "📈 Logistic Regression",
            "⚖️ Ethics & Insights",
            "📁 Project Files",
        ],
        label_visibility="visible"
    )

    st.markdown(f"<hr style='border-color:{BORDER}; margin:8px 0;'>", unsafe_allow_html=True)

    # Live metric pills in sidebar
    best_acc  = max(dt_m["accuracy"], lr_m["accuracy"], rf_m["accuracy"]) * 100
    best_auc  = max(dt_m["roc_auc"],  lr_m["roc_auc"],  rf_m["roc_auc"])
    st.markdown(f"""
    <div style="font-size:11px; color:{SUB_TEXT}; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;">Live Model Stats</div>
    <span class="stat-pill">Best Acc {best_acc:.1f}%</span>
    <span class="stat-pill">AUC {best_auc:.3f}</span>
    <br><br>
    <div style="font-size:11px; color:{SUB_TEXT}; text-align:center;">B.Tech CSE | Machine Learning</div>
    """, unsafe_allow_html=True)

# ============================================================
# HELPER: render metric row
# ============================================================

def metric_row(m):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🎯 Accuracy",  f"{m['accuracy']  * 100:.2f}%")
    c2.metric("🔎 Precision", f"{m['precision'] * 100:.2f}%")
    c3.metric("📡 Recall",    f"{m['recall']    * 100:.2f}%")
    c4.metric("⚖️ F1-Score",  f"{m['f1']        * 100:.2f}%")
    c5.metric("📈 ROC-AUC",   f"{m['roc_auc']:.4f}")

# ============================================================
# 🏠  DASHBOARD
# ============================================================

if page == "🏠 Dashboard":
    st.markdown('<div class="page-content">', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="hero-card">
      <h1 style="color:white; margin:0; font-size:28px;">💄 AI Cosmetic Purchase Prediction</h1>
      <p style="color:rgba(255,255,255,0.85); margin:10px 0 0; font-size:15px;">
        Machine Learning & Business Intelligence Dashboard &nbsp;|&nbsp;
        Decision Tree · Logistic Regression · Random Forest
      </p>
    </div>
    """, unsafe_allow_html=True)

    total     = len(df)
    purchased = int(df["purchased"].sum())
    not_purch = total - purchased
    rate      = purchased / total * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Total Customers",  f"{total:,}")
    c2.metric("🛍️ Purchased",        f"{purchased:,}")
    c3.metric("❌ Not Purchased",     f"{not_purch:,}")
    c4.metric("📊 Conversion Rate",   f"{rate:.2f}%")

    st.markdown(f"<hr style='border-color:{BORDER}'>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📊 Purchase Distribution")
        fig, ax = styled_fig(figsize=(6, 4))
        counts = df["purchased"].value_counts().sort_index()
        colors = [ACCENT2, ACCENT]
        bars = ax.bar(["Not Purchased", "Purchased"],
                      [counts.get(0, 0), counts.get(1, 0)],
                      color=colors, width=0.5, edgecolor=BORDER)
        ax.set_ylabel("Customers")
        ax.set_title("Purchase Outcome", fontweight="bold")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 30,
                    f"{int(h):,}", ha="center", fontweight="bold", color=PLOT_FG)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_b:
        st.subheader("👥 Age Group vs Purchase")
        age_ct = pd.crosstab(df["age_group"], df["purchased"])
        fig, ax = styled_fig(figsize=(7, 4))
        age_ct.plot(kind="bar", ax=ax,
                    color=[f"{ACCENT}88", ACCENT],
                    edgecolor="none")
        ax.set_ylabel("Customers")
        ax.set_xlabel("Age Group")
        ax.set_title("Historical Purchases by Age Group", fontweight="bold")
        ax.legend(["Not Purchased", "Purchased"], framealpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown(f"<hr style='border-color:{BORDER}'>", unsafe_allow_html=True)
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 🔍  EDA EXPLORER
# ============================================================

elif page == "🔍 EDA Explorer":
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.title("🔍 Exploratory Data Analysis")
    st.markdown(
        "Deep-dive into the dataset structure, distributions, correlations, and outliers "
        "before modelling. Understanding the data is the foundation of every ML pipeline."
    )

    st.markdown(f"<hr style='border-color:{BORDER}'>", unsafe_allow_html=True)

    # ── Summary Statistics ──────────────────────────────────
    st.subheader("📋 Dataset Summary Statistics")
    tab_sum, tab_raw = st.tabs(["📊 Numeric Summary", "🗂️ Raw Data (First 50 rows)"])

    with tab_sum:
        numeric_cols = ["tenure", "total", "income", "quantity"]
        st.dataframe(df[numeric_cols].describe().round(2), use_container_width=True)

    with tab_raw:
        st.dataframe(df.head(50), use_container_width=True, hide_index=True)

    st.markdown(f"<hr style='border-color:{BORDER}'>", unsafe_allow_html=True)

    # ── Class Balance ────────────────────────────────────────
    st.subheader("🎯 Target Variable — Purchase Class Balance")
    purchased_cnt = int(df["purchased"].sum())
    not_purch_cnt = len(df) - purchased_cnt
    bal_col1, bal_col2, bal_col3 = st.columns(3)
    bal_col1.metric("✅ Purchased",     f"{purchased_cnt:,}", f"{purchased_cnt/len(df)*100:.1f}%")
    bal_col2.metric("❌ Not Purchased", f"{not_purch_cnt:,}", f"{not_purch_cnt/len(df)*100:.1f}%")
    bal_col3.metric("⚠️ Imbalance Ratio", f"{purchased_cnt/not_purch_cnt:.2f}:1",
                    "Mild — handled by stratified split")

    st.markdown(f"<hr style='border-color:{BORDER}'>", unsafe_allow_html=True)

    # ── Categorical Distributions ────────────────────────────
    st.subheader("📊 Categorical Feature Distributions")
    cat_tab1, cat_tab2, cat_tab3, cat_tab4 = st.tabs(["👤 Age Group", "🚻 Gender", "🗂️ Status", "🌍 Region"])

    def cat_bar(col, title, color1, color2):
        ct = pd.crosstab(df[col], df["purchased"])
        fig, ax = styled_fig(figsize=(7, 4))
        ct.plot(kind="bar", ax=ax, color=[color1, color2], edgecolor="none")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(""); ax.set_ylabel("Customers")
        ax.legend(["Not Purchased", "Purchased"], framealpha=0.3)
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    with cat_tab1:
        cat_bar("age_group", "Purchase by Age Group", f"{ACCENT}55", ACCENT)
        # Purchase rate per age group
        rate_df = df.groupby("age_group")["purchased"].mean().reset_index()
        rate_df.columns = ["Age Group", "Purchase Rate"]
        rate_df["Purchase Rate (%)"] = (rate_df["Purchase Rate"] * 100).round(2)
        st.dataframe(rate_df[["Age Group", "Purchase Rate (%)"]],
                     use_container_width=True, hide_index=True)

    with cat_tab2:
        cat_bar("sex", "Purchase by Gender", f"{ACCENT2}55", ACCENT2)
        gender_rate = df.groupby("sex")["purchased"].mean().reset_index()
        gender_rate["Purchase Rate (%)"] = (gender_rate["purchased"] * 100).round(2)
        st.dataframe(gender_rate[["sex", "Purchase Rate (%)"]],
                     use_container_width=True, hide_index=True)

    with cat_tab3:
        cat_bar("status", "Purchase by Account Status", f"{ACCENT}44", ACCENT)

    with cat_tab4:
        cat_bar("region", "Purchase by Region", f"{ACCENT2}44", ACCENT2)

    st.markdown(f"<hr style='border-color:{BORDER}'>", unsafe_allow_html=True)

    # ── Numeric Distributions ────────────────────────────────
    st.subheader("📈 Numeric Feature Distributions")
    num_features = ["tenure", "total", "income", "quantity"]
    n_col1, n_col2 = st.columns(2)

    for idx, feat in enumerate(num_features):
        with (n_col1 if idx % 2 == 0 else n_col2):
            fig, ax = styled_fig(figsize=(6, 3.5))
            buyers     = df[df["purchased"] == 1][feat]
            non_buyers = df[df["purchased"] == 0][feat]
            ax.hist(non_buyers, bins=30, alpha=0.65, color=ACCENT2,
                    label="Not Purchased", edgecolor="none")
            ax.hist(buyers, bins=30, alpha=0.65, color=ACCENT,
                    label="Purchased", edgecolor="none")
            ax.set_title(f"Distribution of `{feat}`", fontweight="bold")
            ax.set_xlabel(feat.replace("_", " ").title())
            ax.set_ylabel("Frequency")
            ax.legend(framealpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.markdown(f"<hr style='border-color:{BORDER}'>", unsafe_allow_html=True)

    # ── Correlation Matrix ───────────────────────────────────
    st.subheader("🔗 Correlation Matrix (Numeric Features + Target)")
    corr_cols = ["tenure", "total", "income", "quantity", "purchased"]
    corr = df[corr_cols].corr()
    fig, ax = styled_fig(figsize=(7, 5))
    im = ax.imshow(corr, cmap="coolwarm" if st.session_state.dark_mode else "RdYlGn",
                   vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right")
    ax.set_yticklabels(corr_cols)
    ax.set_title("Pearson Correlation Matrix", fontweight="bold")
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if abs(corr.iloc[i, j]) > 0.5 else PLOT_FG)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.markdown(f"""
    <div class="info-card">
      <strong>Key Correlation Findings:</strong>
      <ul style="margin:8px 0 0; padding-left:18px; color:{TEXT};">
        <li><code>total</code> (past spend) has the strongest positive correlation with <code>purchased</code></li>
        <li><code>tenure</code> (loyalty) is a meaningful secondary predictor</li>
        <li><code>income</code> and <code>quantity</code> show moderate correlation with the target</li>
        <li>No strong multicollinearity between features — encodings are safe</li>
      </ul>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"<hr style='border-color:{BORDER}'>", unsafe_allow_html=True)

    # ── Outlier Detection (Box Plots) ────────────────────────
    st.subheader("📦 Outlier Detection — Box Plots")
    fig, axes = styled_fig(1, 4, figsize=(14, 4))
    for i, feat in enumerate(["tenure", "total", "income", "quantity"]):
        bp = axes[i].boxplot(
            [df[df["purchased"]==0][feat], df[df["purchased"]==1][feat]],
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2)
        )
        axes[i].set_xticks([1, 2])
        axes[i].set_xticklabels(["Not\nPurchased", "Purchased"])
        bp["boxes"][0].set_facecolor(f"{ACCENT2}88")
        bp["boxes"][1].set_facecolor(f"{ACCENT}88")
        axes[i].set_title(feat.replace("_"," ").title(), fontweight="bold")
        axes[i].set_ylabel("Value")
    plt.suptitle("Feature Outlier Distribution by Purchase Class", fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close(fig)

    st.markdown(f"""
    <div class="info-card">
      <strong>Outlier Observations:</strong>
      <ul style="margin:8px 0 0; padding-left:18px; color:{TEXT};">
        <li>Buyers show higher median <strong>total spend</strong> and <strong>tenure</strong> — confirming these are the strongest predictors</li>
        <li>Some <code>income</code> outliers exist but do not significantly skew the model (tree-based models are outlier-robust)</li>
        <li>Data cleaning pipeline clips extreme outliers at the 1st–99th percentile</li>
      </ul>
    </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 🎯  BUSINESS RECOMMENDATION
# ============================================================

elif page == "🎯 Business Recommendation":

    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.title("🎯 Age-Group Buyer Potential & Business Recommendation")

    st.markdown(f"""
    <div class="info-card">
      <strong>Core Business Objective:</strong> AI predicts individual purchase probability → 
      we aggregate those probabilities per age group using
      <em>Expected Buyers<sub>k</sub> = Σ P(Purchased = 1 | xᵢ)</em>
      to find which demographic has the highest market demand.
    </div>
    """, unsafe_allow_html=True)

    chosen = st.selectbox(
        "Select Model for Probability Aggregation:",
        ["Logistic Regression (Recommended ✦ Best Accuracy)", "Random Forest (Best Recall)", "Decision Tree"]
    )
    active_pipe = (lr_model if "Logistic" in chosen
                   else rf_model if "Random" in chosen
                   else dt_model)

    full_probas       = active_pipe.predict_proba(X_all)[:, 1]
    df_eval           = df.copy()
    df_eval["purchase_probability"] = full_probas

    age_agg = (
        df_eval.groupby("age_group", observed=False)["purchase_probability"]
        .agg(["count", "sum", "mean"])
        .rename(columns={"count": "Total Customers",
                         "sum":   "Expected Buyers",
                         "mean":  "Avg Probability"})
    )
    total_exp = age_agg["Expected Buyers"].sum()
    age_agg["Buyer Share (%)"]  = ((age_agg["Expected Buyers"] / total_exp) * 100).round(2)
    age_agg["Avg Prob (%)"]     = (age_agg["Avg Probability"] * 100).round(2)
    age_agg["Expected Buyers"]  = age_agg["Expected Buyers"].round(1)
    age_agg = age_agg.sort_values("Buyer Share (%)", ascending=False)

    top_group = age_agg.index[0]
    top_share = age_agg.iloc[0]["Buyer Share (%)"]
    top_exp   = age_agg.iloc[0]["Expected Buyers"]

    st.markdown(f"""
    <div class="success-card">
      <h3 style="margin:0; color:{SUCCESS_BD};">🏆 Primary Marketing Target: {top_group} Age Group</h3>
      <p style="margin:8px 0 0; font-size:15px; color:{TEXT};">
        The <strong>{top_group}</strong> age category represents <strong>{top_share}%</strong> of all predicted buyers 
        (~<strong>{top_exp:,.0f}</strong> expected customers). 
        Concentrate the new unisex product launch budget on this segment.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📋 Expected Buyer Distribution Table")
    st.dataframe(
        age_agg[["Total Customers", "Expected Buyers", "Avg Prob (%)", "Buyer Share (%)"]],
        use_container_width=True
    )

    # Chart
    fig, ax = styled_fig(figsize=(10, 4.5))
    colors_bar = [ACCENT if i == 0 else f"{ACCENT}66" for i in range(len(age_agg))]
    bars = ax.bar(age_agg.index, age_agg["Buyer Share (%)"], color=colors_bar, edgecolor="none")
    ax.set_title(f"Estimated Buyer Share by Age Group — {chosen}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Share of Potential Buyers (%)")
    ax.set_xlabel("Age Group")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.1f}%",
                ha="center", fontweight="bold", color=PLOT_FG)
    ax.set_ylim(0, max(age_agg["Buyer Share (%)"]) + 8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(f"<hr style='border-color:{BORDER}'>", unsafe_allow_html=True)
    st.subheader("⚙️ Decision Threshold Sensitivity Simulator")
    st.caption(
        "Lowering the threshold flags more customers for outreach (higher recall). "
        "Raising it targets only high-confidence buyers (higher precision)."
    )

    threshold     = st.slider("Classification Threshold", 0.10, 0.90, 0.50, 0.05)
    targeted      = (df_eval["purchase_probability"] >= threshold).sum()
    targeted_pct  = targeted / len(df_eval) * 100

    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("🎯 Customers Flagged", f"{targeted:,}", f"{targeted_pct:.1f}% of total")
    tc2.metric("📌 Threshold", f"{threshold:.2f}")
    tc3.metric("💡 Strategy",
               "Broad Outreach" if threshold < 0.40 else
               ("Balanced" if threshold < 0.60 else "Precision Targeting"))

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 🔮  PREDICTION
# ============================================================

elif page == "🔮 Prediction":
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.title("🔮 Customer Purchase Prediction")
    st.markdown("Enter individual customer details to predict purchase likelihood.")

    st.markdown(f"<hr style='border-color:{BORDER}'>", unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Choose ML Model",
        ["Logistic Regression (Recommended ✦ Best Accuracy)", "Random Forest (Best Recall)", "Decision Tree"]
    )
    sel_model = (lr_model if "Logistic" in model_choice
                 else rf_model if "Random" in model_choice
                 else dt_model)

    input_vals = {}
    input_cols = st.columns(2)
    for idx, col in enumerate(X_all.columns):
        with input_cols[idx % 2]:
            if col in CATEGORICAL_COLS:
                opts = sorted(X_all[col].dropna().unique().tolist())
                input_vals[col] = st.selectbox(col.replace("_", " ").title(), opts)
            else:
                input_vals[col] = st.number_input(
                    col.replace("_", " ").title(),
                    value=float(X_all[col].median())
                )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 Predict Purchase Propensity", use_container_width=True):
        inp_df = pd.DataFrame([input_vals])[X_all.columns]
        pred   = sel_model.predict(inp_df)[0]
        prob   = sel_model.predict_proba(inp_df)[0][1]

        st.markdown(f"<hr style='border-color:{BORDER}'>", unsafe_allow_html=True)
        res1, res2 = st.columns(2)
        with res1:
            if pred == 1:
                st.markdown(f"""
                <div class="success-card">
                  <h3 style="color:{SUCCESS_BD}; margin:0;">🛍️ High Propensity</h3>
                  <p style="margin:8px 0 0; color:{TEXT};">This customer is <strong>likely to purchase</strong>!</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-card">
                  <h3 style="color:{WARN_BD}; margin:0;">📉 Low Propensity</h3>
                  <p style="margin:8px 0 0; color:{TEXT};">This customer is <strong>unlikely to purchase</strong>.</p>
                </div>""", unsafe_allow_html=True)
        with res2:
            st.metric("Purchase Probability", f"{prob * 100:.2f}%")
            st.progress(min(max(float(prob), 0.0), 1.0))

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 🤖  MODEL COMPARISON
# ============================================================

elif page == "🤖 Model Comparison":
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.title("🤖 Machine Learning Model Comparison")
    st.markdown("Comprehensive evaluation of all three classification algorithms.")

    # Summary table
    results = pd.DataFrame({
        "Model"         : ["Decision Tree", "Logistic Regression", "Random Forest"],
        "Accuracy (%)"  : [dt_m["accuracy"]*100, lr_m["accuracy"]*100, rf_m["accuracy"]*100],
        "Precision (%)": [dt_m["precision"]*100, lr_m["precision"]*100, rf_m["precision"]*100],
        "Recall (%)"    : [dt_m["recall"]*100, lr_m["recall"]*100, rf_m["recall"]*100],
        "F1-Score (%)"  : [dt_m["f1"]*100, lr_m["f1"]*100, rf_m["f1"]*100],
        "ROC-AUC"       : [dt_m["roc_auc"], lr_m["roc_auc"], rf_m["roc_auc"]],
    }).round(2)

    st.dataframe(results, use_container_width=True, hide_index=True)

    tab1, tab2 = st.tabs(["📊 Metrics Bar Chart", "📉 ROC Curves"])

    with tab1:
        metric_cols = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]
        x = np.arange(len(metric_cols))
        w = 0.25
        fig, ax = styled_fig(figsize=(12, 5))
        ax.bar(x - w, results.iloc[0][metric_cols], w, label="Decision Tree",      color=ACCENT2,    alpha=0.9)
        ax.bar(x,     results.iloc[1][metric_cols], w, label="Logistic Regression", color="#06b6d4",  alpha=0.9)
        ax.bar(x + w, results.iloc[2][metric_cols], w, label="Random Forest",       color=ACCENT,     alpha=0.9)
        ax.set_ylabel("Score (%)");  ax.set_ylim(0, 105)
        ax.set_title("Model Evaluation Metrics Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x);  ax.set_xticklabels(metric_cols)
        ax.legend(framealpha=0.3);  ax.grid(axis="y", alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True);  plt.close(fig)

    with tab2:
        fig, ax = styled_fig(figsize=(9, 5))
        for model_name, m, color in [
            ("Decision Tree",      dt_m, ACCENT2),
            ("Logistic Regression",lr_m, "#06b6d4"),
            ("Random Forest",      rf_m, ACCENT),
        ]:
            fpr, tpr, _ = roc_curve(y_test, m["probas"])
            ax.plot(fpr, tpr, label=f"{model_name} (AUC={m['roc_auc']:.3f})", color=color, lw=2)
        ax.plot([0, 1], [0, 1], ":", color=SUB_TEXT, lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
        ax.legend(framealpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True);  plt.close(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 🌳  DECISION TREE
# ============================================================

elif page == "🌳 Decision Tree":
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.title("🌳 Decision Tree Classifier")
    st.markdown("Interpretable rule-based classification with `max_depth=5`.")

    metric_row(dt_m)
    st.markdown(f"<hr style='border-color:{BORDER}'>", unsafe_allow_html=True)

    tab_tree, tab_cm, tab_roc = st.tabs(["🌳 Tree Structure", "🔢 Confusion Matrix", "📉 ROC Curve"])

    with tab_tree:
        st.subheader("Decision Tree – First 3 Levels")
        trained_tree = dt_model.named_steps["decision_tree"]
        feat_names   = dt_model.named_steps["preprocessor"].get_feature_names_out()
        fig, ax = plt.subplots(figsize=(24, 11), facecolor=PLOT_BG)
        ax.set_facecolor(PLOT_BG)
        plot_tree(trained_tree, feature_names=feat_names,
                  class_names=["Not Purchased", "Purchased"],
                  filled=True, max_depth=3, fontsize=10, rounded=True, ax=ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True);  plt.close(fig)

    with tab_cm:
        cm = confusion_matrix(y_test, dt_m["preds"])
        fig, ax = styled_fig(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Purples")
        plt.colorbar(im, ax=ax)
        ax.set_title("Decision Tree – Confusion Matrix", fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Not Purchased", "Purchased"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Not Purchased", "Purchased"])
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else PLOT_FG
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        fontsize=18, fontweight="bold", color=color)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True);  plt.close(fig)

    with tab_roc:
        fpr, tpr, _ = roc_curve(y_test, dt_m["probas"])
        fig, ax = styled_fig(figsize=(7, 4))
        ax.plot(fpr, tpr, color=ACCENT2, lw=2.5,
                label=f"Decision Tree (AUC={dt_m['roc_auc']:.4f})")
        ax.fill_between(fpr, tpr, alpha=0.12, color=ACCENT2)
        ax.plot([0, 1], [0, 1], ":", color=SUB_TEXT)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Decision Tree – ROC Curve", fontweight="bold")
        ax.legend(framealpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True);  plt.close(fig)

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 🌲  RANDOM FOREST
# ============================================================

elif page == "🌲 Random Forest":
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.title("🌲 Random Forest Classifier")
    st.markdown(
        "Ensemble of **300 trees** with GridSearch-optimised hyperparameters "
        "(`max_depth=8`, `min_samples_split=5`, `max_features='sqrt'`)."
    )

    metric_row(rf_m)
    st.markdown(f"<hr style='border-color:{BORDER}'>", unsafe_allow_html=True)

    tab_fi, tab_cm, tab_roc = st.tabs(["📌 Feature Importance", "🔢 Confusion Matrix", "📉 ROC Curve"])

    with tab_fi:
        feat_names  = rf_model.named_steps["preprocessor"].get_feature_names_out()
        importances = rf_model.named_steps["random_forest"].feature_importances_
        imp_df = (
            pd.DataFrame({"Feature": feat_names, "Importance": importances})
            .sort_values("Importance", ascending=False)
        )

        col_f1, col_f2 = st.columns([1, 1.6])
        with col_f1:
            st.dataframe(imp_df.head(12).reset_index(drop=True), use_container_width=True)
        with col_f2:
            top10 = imp_df.head(10).sort_values("Importance")
            fig, ax = styled_fig(figsize=(8, 5))
            bars = ax.barh(top10["Feature"], top10["Importance"], color=ACCENT, alpha=0.9)
            ax.set_xlabel("Importance Score")
            ax.set_title("Top 10 Feature Importances – Random Forest", fontweight="bold")
            for b in bars:
                ax.text(b.get_width() + 0.001, b.get_y() + b.get_height() / 2,
                        f"{b.get_width():.3f}", va="center", fontsize=9, color=PLOT_FG)
            ax.set_xlim(0, float(top10["Importance"].max()) * 1.18)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True);  plt.close(fig)

    with tab_cm:
        cm = confusion_matrix(y_test, rf_m["preds"])
        fig, ax = styled_fig(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Greens")
        plt.colorbar(im, ax=ax)
        ax.set_title("Random Forest – Confusion Matrix", fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Not Purchased", "Purchased"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Not Purchased", "Purchased"])
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else PLOT_FG
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        fontsize=18, fontweight="bold", color=color)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True);  plt.close(fig)

    with tab_roc:
        fpr, tpr, _ = roc_curve(y_test, rf_m["probas"])
        fig, ax = styled_fig(figsize=(7, 4))
        ax.plot(fpr, tpr, color=ACCENT, lw=2.5,
                label=f"Random Forest (AUC={rf_m['roc_auc']:.4f})")
        ax.fill_between(fpr, tpr, alpha=0.12, color=ACCENT)
        ax.plot([0, 1], [0, 1], ":", color=SUB_TEXT)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Random Forest – ROC Curve", fontweight="bold")
        ax.legend(framealpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True);  plt.close(fig)

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 📈  LOGISTIC REGRESSION
# ============================================================

elif page == "📈 Logistic Regression":
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.title("📈 Logistic Regression Model")
    st.markdown("Probabilistic linear baseline using sigmoid log-odds function (`max_iter=1000`).")

    metric_row(lr_m)
    st.markdown(f"<hr style='border-color:{BORDER}'>", unsafe_allow_html=True)

    tab_cm, tab_roc = st.tabs(["🔢 Confusion Matrix", "📉 ROC Curve"])

    with tab_cm:
        cm = confusion_matrix(y_test, lr_m["preds"])
        fig, ax = styled_fig(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_title("Logistic Regression – Confusion Matrix", fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Not Purchased", "Purchased"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Not Purchased", "Purchased"])
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else PLOT_FG
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        fontsize=18, fontweight="bold", color=color)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True);  plt.close(fig)


    with tab_roc:
        fpr, tpr, _ = roc_curve(y_test, lr_m["probas"])
        fig, ax = styled_fig(figsize=(7, 4))
        ax.plot(fpr, tpr, color=ACCENT, lw=2.5,
                label=f"LR (AUC={lr_m['roc_auc']:.4f})")
        ax.fill_between(fpr, tpr, alpha=0.12, color=ACCENT)
        ax.plot([0, 1], [0, 1], ":", color=SUB_TEXT)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Logistic Regression – ROC Curve", fontweight="bold")
        ax.legend(framealpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True);  plt.close(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# ⚖️  ETHICS & INSIGHTS
# ============================================================

elif page == "⚖️ Ethics & Insights":
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.title("⚖️ Ethics, Responsible AI & Unisex Insights")

    st.markdown(f"""
    <div class="warning-card">
      <h4 style="margin:0; color:{WARN_BD};">⚠️ Purchase Propensity vs. Stereotyping</h4>
      <p style="margin:8px 0 0; color:{TEXT};">
        This project models <strong>purchase propensity</strong> — the likelihood of a commercial transaction 
        based on historical engagement — <strong>NOT</strong> a value judgment about any demographic group.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.subheader("1. Unisex Skincare Market Dynamics")
        st.markdown(f"""
        <div class="info-card">
          <ul style="margin:0; padding-left:18px; color:{TEXT};">
            <li><strong>Breaking Stigmas:</strong> Historical skincare marketing disproportionately targeted women. 
                Unisex formulations create neutral, health-first positioning.</li>
            <li><strong>Cross-Demographic Adoption:</strong> Younger male cohorts (18–35) show significantly 
                higher purchase propensity for unisex skincare vs. older cohorts.</li>
            <li><strong>Inclusive Messaging:</strong> Marketing should emphasize dermatological health and hydration 
                rather than anti-aging panic or gender-conforming beauty standards.</li>
          </ul>
        </div>""", unsafe_allow_html=True)

    with col_e2:
        st.subheader("2. Data Privacy & Governance")
        st.markdown(f"""
        <div class="info-card">
          <ul style="margin:0; padding-left:18px; color:{TEXT};">
            <li><strong>Consent & Transparency:</strong> Customer data gathered only with explicit opt-in 
                for product recommendation and analytics.</li>
            <li><strong>No Algorithmic Redlining:</strong> Models must not systematically exclude entire 
                regions or communities from receiving product promotions.</li>
            <li><strong>Fairness Monitoring:</strong> Real-world deployment requires continuous bias audits 
                and periodic retraining on representative data (GDPR / DPDP compliant).</li>
          </ul>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"<hr style='border-color:{BORDER}'>", unsafe_allow_html=True)
    st.subheader("3. Gender Breakdown — Purchase Propensity")
    gender_pur = df.groupby("sex")["purchased"].mean() * 100
    fig, ax = styled_fig(figsize=(6, 3.5))
    bars = ax.bar(gender_pur.index, gender_pur.values,
                  color=[ACCENT, ACCENT2], width=0.4, edgecolor="none")
    ax.set_ylabel("Avg Purchase Rate (%)")
    ax.set_title("Purchase Rate by Gender", fontweight="bold")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", fontweight="bold", color=PLOT_FG)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True);  plt.close(fig)

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 📁  PROJECT FILES
# ============================================================

elif page == "📁 Project Files":
    st.markdown('<div class="page-content">', unsafe_allow_html=True)
    st.title("📁 Project Files")
    st.markdown("Browse, preview, and download all project source files.")

    folders = ["data", "models", "notebooks", "reports", "docs", "src", "visualisations"]
    for folder_name in folders:
        folder_path = BASE_DIR / folder_name
        if not folder_path.exists():
            continue
        files = [
            f for f in folder_path.rglob("*")
            if f.is_file() and "__pycache__" not in f.parts and not f.name.endswith(".pyc")
        ]
        if not files:
            continue
        st.subheader(f"📂 {folder_name}/")
        for fp in files:
            rel = fp.relative_to(BASE_DIR)
            with st.expander(f"📄 {rel}"):
                ext = fp.suffix.lower()
                if ext in [".py", ".ipy"]:
                    st.code(fp.read_text(encoding="utf-8", errors="replace"), language="python")
                elif ext == ".csv":
                    st.dataframe(pd.read_csv(fp).head(30), use_container_width=True)
                elif ext in [".md", ".txt"]:
                    st.markdown(fp.read_text(encoding="utf-8", errors="replace"))
                elif ext in [".png", ".jpg", ".jpeg"]:
                    st.image(str(fp), use_container_width=True)
                try:
                    with open(fp, "rb") as f:
                        clean_key = f"dl_{str(rel).replace(chr(92), '_').replace('/', '_')}"
                        st.download_button(
                            f"⬇️ Download {fp.name}", f.read(),
                            file_name=fp.name, key=clean_key
                        )
                except Exception:
                    pass

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown(f"<hr style='border-color:{BORDER}; margin-top:3rem;'>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center; color:{SUB_TEXT}; font-size:12px; padding-bottom:1rem;">
  💄 AI Cosmetic Purchase Prediction &nbsp;|&nbsp;
  Decision Tree · Logistic Regression · Random Forest (GridSearch Optimised) &nbsp;|&nbsp;
  B.Tech CSE Machine Learning Project
</div>
""", unsafe_allow_html=True)