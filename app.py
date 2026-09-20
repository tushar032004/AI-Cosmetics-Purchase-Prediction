# ============================================================
# AI COSMETIC PURCHASE PREDICTION
# STREAMLIT APPLICATION
# ============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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
        padding-left: 3rem;
        padding-right: 3rem;
    }

    h1 {
        font-weight: 700;
        letter-spacing: -1px;
    }

    h2 {
        font-weight: 650;
        margin-top: 2rem;
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
        padding: 20px;
        border-radius: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 45px;
    }

    .stDownloadButton > button {
        border-radius: 8px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    .streamlit-expanderHeader {
        font-weight: 600;
    }

    hr {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "cleaned_cosmetics_dataset.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(CSV_PATH)


try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "Dataset not found. Please check that the following file exists:"
    )

    st.code(
        str(CSV_PATH)
    )

    st.stop()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Id",
    "purchased",
    "sex",
    "age_group",
    "status",
    "region"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "The following required columns are missing:"
    )

    st.write(missing_columns)

    st.stop()


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(
    ["Id", "purchased"],
    axis=1
)

y = df["purchased"]


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

categorical_columns = [
    "sex",
    "age_group",
    "status",
    "region"
]


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# DECISION TREE MODEL
# ============================================================

decision_tree_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "decision_tree",
            DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            )
        )
    ]
)


decision_tree_model.fit(
    X_train,
    y_train
)


decision_tree_pred = decision_tree_model.predict(
    X_test
)


decision_tree_probability = (
    decision_tree_model.predict_proba(
        X_test
    )[:, 1]
)


decision_tree_accuracy = accuracy_score(
    y_test,
    decision_tree_pred
)

decision_tree_precision = precision_score(
    y_test,
    decision_tree_pred,
    zero_division=0
)

decision_tree_recall = recall_score(
    y_test,
    decision_tree_pred,
    zero_division=0
)

decision_tree_f1 = f1_score(
    y_test,
    decision_tree_pred,
    zero_division=0
)


# ============================================================
# LOGISTIC REGRESSION MODEL
# ============================================================

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


logistic_model.fit(
    X_train,
    y_train
)


logistic_pred = logistic_model.predict(
    X_test
)


logistic_probability = (
    logistic_model.predict_proba(
        X_test
    )[:, 1]
)


logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

logistic_precision = precision_score(
    y_test,
    logistic_pred,
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    logistic_pred,
    zero_division=0
)

logistic_f1 = f1_score(
    y_test,
    logistic_pred,
    zero_division=0
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        # 💄 Cosmetic AI

        **Purchase Prediction System**
        """
    )

    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "🏠 Dashboard",
            "🔮 Prediction",
            "🤖 Models",
            "🌳 Decision Tree",
            "📈 Logistic Regression",
            "📁 Project Files"
        ]
    )

    st.divider()

    st.caption(
        "AI Cosmetic Purchase Prediction"
    )

    st.caption(
        "Machine Learning Project"
    )


# ============================================================
# 🏠 DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title(
        "💄 AI Cosmetic Purchase Prediction"
    )

    st.markdown(
        """
        ### Machine Learning Dashboard

        This application predicts whether a customer is likely
        to purchase a cosmetic product using machine learning.
        """
    )

    st.divider()

    # ========================================================
    # KPI CARDS
    # ========================================================

    total_customers = len(df)

    total_purchased = int(
        df["purchased"].sum()
    )

    total_not_purchased = (
        total_customers
        - total_purchased
    )

    purchase_rate = (
        total_purchased
        / total_customers
        * 100
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "👥 Total Customers",
            f"{total_customers:,}"
        )

    with col2:

        st.metric(
            "🛍️ Purchased",
            f"{total_purchased:,}"
        )

    with col3:

        st.metric(
            "❌ Not Purchased",
            f"{total_not_purchased:,}"
        )

    with col4:

        st.metric(
            "📊 Purchase Rate",
            f"{purchase_rate:.2f}%"
        )

    st.divider()

    # ========================================================
    # PURCHASE DISTRIBUTION
    # ========================================================

    st.subheader(
        "📊 Purchase Distribution"
    )

    purchase_counts = (
        df["purchased"]
        .value_counts()
        .sort_index()
    )

    labels = [
        "Not Purchased",
        "Purchased"
    ]

    values = [
        purchase_counts.get(0, 0),
        purchase_counts.get(1, 0)
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    bars = ax.bar(
        labels,
        values
    )

    ax.set_title(
        "Cosmetic Purchase Distribution",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Purchase Status",
        fontsize=12
    )

    ax.set_ylabel(
        "Number of Customers",
        fontsize=12
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.divider()

    # ========================================================
    # AGE GROUP ANALYSIS
    # ========================================================

    st.subheader(
        "👥 Purchase Behaviour by Age Group"
    )

    age_purchase = pd.crosstab(
        df["age_group"],
        df["purchased"]
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    age_purchase.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Purchase Behaviour by Age Group",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Age Group",
        fontsize=12
    )

    ax.set_ylabel(
        "Number of Customers",
        fontsize=12
    )

    ax.legend(
        [
            "Not Purchased",
            "Purchased"
        ],
        title="Purchase Status"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.divider()

    # ========================================================
    # REGION ANALYSIS
    # ========================================================

    st.subheader(
        "🌍 Purchase Behaviour by Region"
    )

    region_purchase = pd.crosstab(
        df["region"],
        df["purchased"]
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    region_purchase.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Purchase Behaviour by Region",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Region",
        fontsize=12
    )

    ax.set_ylabel(
        "Number of Customers",
        fontsize=12
    )

    ax.legend(
        [
            "Not Purchased",
            "Purchased"
        ],
        title="Purchase Status"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# 🔮 PREDICTION
# ============================================================

elif page == "🔮 Prediction":

    st.title(
        "🔮 Cosmetic Purchase Prediction"
    )

    st.markdown(
        """
        Enter customer information and select a machine
        learning model to predict the purchase outcome.
        """
    )

    st.divider()

    # ========================================================
    # MODEL SELECTION
    # ========================================================

    st.subheader(
        "🤖 Select Machine Learning Model"
    )

    model_choice = st.selectbox(
        "Choose a model",
        [
            "Decision Tree",
            "Logistic Regression"
        ]
    )

    st.divider()

    # ========================================================
    # INPUT FORM
    # ========================================================

    st.subheader(
        "👤 Customer Information"
    )

    input_values = {}

    input_columns = st.columns(2)

    feature_columns = [
        column
        for column in X.columns
    ]

    for index, column in enumerate(
        feature_columns
    ):

        with input_columns[index % 2]:

            if column in categorical_columns:

                options = (
                    sorted(
                        X[column]
                        .dropna()
                        .unique()
                        .tolist()
                    )
                )

                input_values[column] = st.selectbox(
                    column.replace(
                        "_",
                        " "
                    ).title(),
                    options
                )

            else:

                input_values[column] = st.number_input(
                    column.replace(
                        "_",
                        " "
                    ).title(),
                    value=float(
                        X[column].median()
                    )
                )

    st.divider()

    # ========================================================
    # CREATE INPUT DATAFRAME
    # ========================================================

    input_df = pd.DataFrame(
        [input_values]
    )

    # Make sure columns are in same order
    input_df = input_df[
        X.columns
    ]

    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    if st.button(
        "🔮 Predict Purchase",
        use_container_width=True
    ):

        if model_choice == "Decision Tree":

            selected_model = (
                decision_tree_model
            )

        else:

            selected_model = (
                logistic_model
            )

        # Prediction
        prediction = (
            selected_model.predict(
                input_df
            )[0]
        )

        # Probability
        probability = (
            selected_model
            .predict_proba(
                input_df
            )[0][1]
        )

        st.divider()

        # ====================================================
        # RESULT
        # ====================================================

        st.subheader(
            "📋 Prediction Result"
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if prediction == 1:

                st.success(
                    "🛍️ Customer is likely to purchase."
                )

            else:

                st.info(
                    "ℹ️ Customer is unlikely to purchase."
                )

        with result_col2:

            st.metric(
                "Purchase Probability",
                f"{probability * 100:.2f}%"
            )

        st.progress(
            float(probability)
        )

        st.caption(
            f"Model used: {model_choice}"
        )


# ============================================================
# 🤖 MODELS
# ============================================================

elif page == "🤖 Models":

    st.title(
        "🤖 Machine Learning Models"
    )

    st.markdown(
        """
        Compare the performance of the machine learning
        models used for cosmetic purchase prediction.
        """
    )

    st.divider()

    # ========================================================
    # MODEL RESULTS
    # ========================================================

    results = pd.DataFrame({

        "Model": [
            "Decision Tree",
            "Logistic Regression"
        ],

        "Accuracy": [
            decision_tree_accuracy,
            logistic_accuracy
        ],

        "Precision": [
            decision_tree_precision,
            logistic_precision
        ],

        "Recall": [
            decision_tree_recall,
            logistic_recall
        ],

        "F1 Score": [
            decision_tree_f1,
            logistic_f1
        ]

    })

    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Decision Tree",
            f"{decision_tree_accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Logistic Regression",
            f"{logistic_accuracy * 100:.2f}%"
        )

    with col3:

        st.metric(
            "DT F1 Score",
            f"{decision_tree_f1 * 100:.2f}%"
        )

    with col4:

        st.metric(
            "LR F1 Score",
            f"{logistic_f1 * 100:.2f}%"
        )

    st.divider()

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    st.subheader(
        "📋 Performance Summary"
    )

    display_results = results.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]:

        display_results[column] = (
            display_results[column] * 100
        ).round(2)

    display_results = display_results.rename(
        columns={
            "Accuracy": "Accuracy (%)",
            "Precision": "Precision (%)",
            "Recall": "Recall (%)",
            "F1 Score": "F1 Score (%)"
        }
    )

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ========================================================
    # ACCURACY GRAPH
    # ========================================================

    st.subheader(
        "📈 Accuracy Comparison"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    accuracy_values = (
        results["Accuracy"] * 100
    )

    bars = ax.bar(
        results["Model"],
        accuracy_values
    )

    ax.set_title(
        "Machine Learning Model Accuracy",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Model",
        fontsize=12
    )

    ax.set_ylabel(
        "Accuracy (%)",
        fontsize=12
    )

    ax.set_ylim(
        0,
        100
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    for bar in bars:

        height = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            height + 1,
            f"{height:.2f}%",
            ha="center",
            fontsize=11,
            fontweight="bold"
        )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.divider()

    # ========================================================
    # ALL METRICS GRAPH
    # ========================================================

    st.subheader(
        "📊 Model Metrics Comparison"
    )

    metrics_to_plot = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    x = range(
        len(metrics_to_plot)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    dt_values = [
        decision_tree_accuracy * 100,
        decision_tree_precision * 100,
        decision_tree_recall * 100,
        decision_tree_f1 * 100
    ]

    lr_values = [
        logistic_accuracy * 100,
        logistic_precision * 100,
        logistic_recall * 100,
        logistic_f1 * 100
    ]

    bars1 = ax.bar(
        [
            i - width / 2
            for i in x
        ],
        dt_values,
        width,
        label="Decision Tree"
    )

    bars2 = ax.bar(
        [
            i + width / 2
            for i in x
        ],
        lr_values,
        width,
        label="Logistic Regression"
    )

    ax.set_title(
        "Model Performance Metrics",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Metric",
        fontsize=12
    )

    ax.set_ylabel(
        "Score (%)",
        fontsize=12
    )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        metrics_to_plot
    )

    ax.set_ylim(
        0,
        100
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    for bars in [
        bars1,
        bars2
    ]:

        for bar in bars:

            height = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                height + 1,
                f"{height:.1f}",
                ha="center",
                fontsize=9
            )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# 🌳 DECISION TREE
# ============================================================

elif page == "🌳 Decision Tree":

    st.title(
        "🌳 Decision Tree"
    )

    st.markdown(
        """
        Visualisation and evaluation of the Decision Tree
        classification model.
        """
    )

    st.divider()

    # ========================================================
    # PERFORMANCE
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            f"{decision_tree_accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Precision",
            f"{decision_tree_precision * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Recall",
            f"{decision_tree_recall * 100:.2f}%"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{decision_tree_f1 * 100:.2f}%"
        )

    st.divider()

    # ========================================================
    # TREE VISUALIZATION
    # ========================================================

    st.subheader(
        "🌳 Decision Tree Visualization"
    )

    trained_tree = (
        decision_tree_model
        .named_steps[
            "decision_tree"
        ]
    )

    trained_preprocessor = (
        decision_tree_model
        .named_steps[
            "preprocessor"
        ]
    )

    # Get feature names
    try:

        feature_names = (
            trained_preprocessor
            .get_feature_names_out()
        )

    except:

        feature_names = None

    fig, ax = plt.subplots(
        figsize=(30, 16)
    )

    plot_tree(
        trained_tree,
        feature_names=feature_names,
        class_names=[
            "Not Purchased",
            "Purchased"
        ],
        filled=True,
        max_depth=3,
        fontsize=12,
        rounded=True,
        proportion=True,
        ax=ax
    )

    ax.set_title(
        "Decision Tree - First 3 Levels",
        fontsize=18,
        fontweight="bold"
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.info(
        "Only the first 3 levels are displayed to keep "
        "the tree readable."
    )

    st.divider()

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.subheader(
        "🔢 Confusion Matrix"
    )

    cm = confusion_matrix(
        y_test,
        decision_tree_pred
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.imshow(cm)

    ax.set_title(
        "Decision Tree Confusion Matrix",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Predicted Label",
        fontsize=12
    )

    ax.set_ylabel(
        "Actual Label",
        fontsize=12
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        [
            "Not Purchased",
            "Purchased"
        ]
    )

    ax.set_yticklabels(
        [
            "Not Purchased",
            "Purchased"
        ]
    )

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold"
            )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.divider()

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.subheader(
        "📌 Feature Importance"
    )

    importances = (
        trained_tree.feature_importances_
    )

    if feature_names is not None:

        importance_df = pd.DataFrame({

            "Feature": feature_names,

            "Importance": importances

        })

        importance_df = (
            importance_df
            .sort_values(
                "Importance",
                ascending=False
            )
        )

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True
        )

        # Graph
        top_features = (
            importance_df
            .head(10)
            .sort_values(
                "Importance"
            )
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.barh(
            top_features["Feature"],
            top_features["Importance"]
        )

        ax.set_title(
            "Top 10 Feature Importances",
            fontsize=16,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Importance",
            fontsize=12
        )

        ax.set_ylabel(
            "Feature",
            fontsize=12
        )

        ax.grid(
            axis="x",
            alpha=0.3
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


# ============================================================
# 📈 LOGISTIC REGRESSION
# ============================================================

elif page == "📈 Logistic Regression":

    st.title(
        "📈 Logistic Regression"
    )

    st.markdown(
        """
        Logistic Regression predicts whether a customer
        is likely to purchase a cosmetic product.
        """
    )

    st.divider()

    # ========================================================
    # PERFORMANCE METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            f"{logistic_accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Precision",
            f"{logistic_precision * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Recall",
            f"{logistic_recall * 100:.2f}%"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{logistic_f1 * 100:.2f}%"
        )

    st.divider()

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.subheader(
        "🔢 Confusion Matrix"
    )

    cm = confusion_matrix(
        y_test,
        logistic_pred
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.imshow(cm)

    ax.set_title(
        "Logistic Regression Confusion Matrix",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Predicted Label",
        fontsize=12
    )

    ax.set_ylabel(
        "Actual Label",
        fontsize=12
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        [
            "Not Purchased",
            "Purchased"
        ]
    )

    ax.set_yticklabels(
        [
            "Not Purchased",
            "Purchased"
        ]
    )

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold"
            )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.divider()

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    st.subheader(
        "📋 Classification Report"
    )

    report = classification_report(
        y_test,
        logistic_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    st.divider()

    # ========================================================
    # LOGISTIC REGRESSION INFORMATION
    # ========================================================

    st.subheader(
        "ℹ️ Model Information"
    )

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.info(
            """
            **Algorithm:** Logistic Regression

            **Maximum Iterations:** 1000

            **Target:** Customer Purchase
            """
        )

    with info_col2:

        st.info(
            """
            **Problem Type:** Binary Classification

            **Output:** Purchased / Not Purchased

            **Preprocessing:** One-Hot Encoding
            """
        )


# ============================================================
# 📁 PROJECT FILES
# ============================================================

elif page == "📁 Project Files":

    st.title(
        "📁 Project Files"
    )

    st.markdown(
        """
        Browse the files included in the AI Cosmetic
        Purchase Prediction project.
        """
    )

    st.divider()

    folders = [
        "data",
        "models",
        "notebooks",
        "reports",
        "visualisations"
    ]

    for folder_name in folders:

        folder_path = (
            BASE_DIR
            / folder_name
        )

        st.subheader(
            f"📂 {folder_name}"
        )

        if not folder_path.exists():

            st.warning(
                f"{folder_name} folder not found."
            )

            continue

        files = [
            file
            for file in folder_path.rglob("*")
            if file.is_file()
        ]

        if not files:

            st.info(
                "No files found."
            )

            continue

        for file_path in files:

            relative_path = (
                file_path.relative_to(
                    BASE_DIR
                )
            )

            with st.expander(
                f"📄 {relative_path}"
            ):

                file_extension = (
                    file_path
                    .suffix
                    .lower()
                )

                # --------------------------------------------
                # PYTHON FILE
                # --------------------------------------------

                if file_extension == ".py":

                    try:

                        content = (
                            file_path
                            .read_text(
                                encoding="utf-8"
                            )
                        )

                        st.code(
                            content,
                            language="python"
                        )

                    except Exception as error:

                        st.error(
                            f"Unable to read file: {error}"
                        )

                # --------------------------------------------
                # CSV FILE
                # --------------------------------------------

                elif file_extension == ".csv":

                    try:

                        preview_df = pd.read_csv(
                            file_path
                        )

                        st.dataframe(
                            preview_df.head(100),
                            use_container_width=True
                        )

                    except Exception as error:

                        st.error(
                            f"Unable to read CSV: {error}"
                        )

                # --------------------------------------------
                # TEXT / MARKDOWN
                # --------------------------------------------

                elif file_extension in [
                    ".txt",
                    ".md"
                ]:

                    try:

                        content = (
                            file_path
                            .read_text(
                                encoding="utf-8"
                            )
                        )

                        st.text(
                            content
                        )

                    except Exception as error:

                        st.error(
                            f"Unable to read file: {error}"
                        )

                # --------------------------------------------
                # IMAGE
                # --------------------------------------------

                elif file_extension in [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp"
                ]:

                    st.image(
                        str(file_path),
                        use_container_width=True
                    )

                # --------------------------------------------
                # OTHER FILE
                # --------------------------------------------

                else:

                    st.write(
                        f"File type: {file_extension}"
                    )

                # --------------------------------------------
                # DOWNLOAD BUTTON
                # --------------------------------------------

                try:

                    with open(
                        file_path,
                        "rb"
                    ) as file:

                        st.download_button(
                            label="⬇️ Download File",
                            data=file.read(),
                            file_name=file_path.name,
                            key=f"download_{relative_path}"
                        )

                except Exception:

                    pass


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "💄 AI Cosmetic Purchase Prediction | "
    "Decision Tree + Logistic Regression"
)