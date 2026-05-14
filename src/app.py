"""
Pokémon Legendary KNN Classifier - Streamlit Mini-System Prototype
Data Mining and Business Intelligence | Special Laboratory Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pokémon Legendary KNN Classifier",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Nunito:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }

    .main-title {
        font-family: 'Press Start 2P', cursive;
        font-size: 1.4rem;
        color: #FFCB05;
        text-shadow: 3px 3px 0px #2A75BB;
        text-align: center;
        padding: 1rem 0 0.5rem;
        letter-spacing: 1px;
    }
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-family: 'Press Start 2P', cursive;
        font-size: 0.75rem;
        color: #2A75BB;
        border-bottom: 3px solid #FFCB05;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
        letter-spacing: 1px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #FFCB05;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        color: white;
    }
    .metric-value {
        font-family: 'Press Start 2P', cursive;
        font-size: 1.4rem;
        color: #FFCB05;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #aaa;
        margin-top: 0.3rem;
    }
    .legendary-badge {
        background: linear-gradient(135deg, #FFCB05, #FF6B00);
        color: #1a1a2e;
        font-weight: 800;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
    }
    .normal-badge {
        background: linear-gradient(135deg, #2A75BB, #1a5c9e);
        color: white;
        font-weight: 800;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
    }
    .neighbor-card {
        background: #f8f9ff;
        border-left: 4px solid #2A75BB;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .explanation-box {
        background: linear-gradient(135deg, #fff9e6, #fff3cd);
        border: 2px dashed #FFCB05;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .stButton > button {
        background: linear-gradient(135deg, #FFCB05, #FF6B00);
        color: #1a1a2e;
        font-weight: 800;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-family: 'Nunito', sans-serif;
        font-size: 1rem;
        transition: transform 0.1s;
    }
    .stButton > button:hover { transform: scale(1.03); }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e, #16213e); }
    div[data-testid="stSidebar"] * { color: #eee !important; }
</style>
""", unsafe_allow_html=True)

# ── KNN Helper Functions ──────────────────────────────────────────────────────
def euclidean_distance(a, b):
    """Compute Euclidean distance between two numpy arrays."""
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict_with_neighbors(X_train, y_train, query, k):
    """
    Manual KNN prediction that also returns the K nearest neighbors
    with their distances and classes.
    """
    distances = [euclidean_distance(query, x) for x in X_train]
    sorted_indices = np.argsort(distances)[:k]
    neighbors = [
        {"rank": i + 1,
         "index": int(sorted_indices[i]),
         "distance": round(distances[sorted_indices[i]], 4),
         "class": y_train.iloc[sorted_indices[i]] if hasattr(y_train, 'iloc') else y_train[sorted_indices[i]]}
        for i in range(k)
    ]
    classes = [n["class"] for n in neighbors]
    predicted = max(set(classes), key=classes.count)
    return predicted, neighbors


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
for key in ["df", "X_train", "X_test", "y_train", "y_test",
            "scaler", "feature_cols", "model", "trained"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "trained" not in st.session_state:
    st.session_state["trained"] = False


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ Navigation")
    page = st.radio("Go to", [
        "🏠 Home",
        "📂 Dataset Preview",
        "⚙️ Model Setup & Train",
        "📊 Evaluation Results",
        "🔮 New Record Prediction",
        "📈 Visualizations"
    ])
    st.markdown("---")
    st.markdown("**Project:** KNN Classifier Prototype")
    st.markdown("**Algorithm:** K-Nearest Neighbor")
    st.markdown("**Dataset:** Pokémon Base Stats")
    st.markdown("**Target:** Legendary Status")
    if st.session_state["trained"]:
        st.success("✅ Model is trained!")
    else:
        st.warning("⚠️ Model not trained yet")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<div class="main-title">⚡ POKÉMON LEGENDARY CLASSIFIER ⚡</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">K-Nearest Neighbor Mini-System Prototype | Data Mining & Business Intelligence</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="section-header">WHAT THIS SYSTEM DOES</div>', unsafe_allow_html=True)
        st.markdown("""
        This mini-system uses the **K-Nearest Neighbor (KNN) algorithm** to classify a Pokémon
        as **Legendary** or **Non-Legendary** based on its base battle statistics.

        The KNN algorithm works by:
        1. **Computing distances** from a new Pokémon to all training Pokémon
        2. **Finding the K closest** (nearest) Pokémon
        3. **Voting** — the majority class among those K neighbors wins

        > *"Tell me who your neighbors are, and I'll tell you who you are."*
        """)

        st.markdown('<div class="section-header">HOW TO USE THIS APP</div>', unsafe_allow_html=True)
        st.markdown("""
        | Step | Page | Action |
        |------|------|--------|
        | 1 | 📂 Dataset Preview | Upload your CSV and inspect the data |
        | 2 | ⚙️ Model Setup & Train | Select features, K value, split ratio |
        | 3 | 📊 Evaluation Results | View confusion matrix and metrics |
        | 4 | 🔮 New Record Prediction | Classify a brand new Pokémon |
        | 5 | 📈 Visualizations | Explore charts and distributions |
        """)

    with col2:
        st.markdown('<div class="section-header">QUICK STATS</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">KNN</div>
            <div class="metric-label">Algorithm</div>
        </div><br>
        <div class="metric-card">
            <div class="metric-value">6</div>
            <div class="metric-label">Base Stat Features</div>
        </div><br>
        <div class="metric-card">
            <div class="metric-value">2</div>
            <div class="metric-label">Target Classes</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Target Classes:**")
        st.markdown('<span class="legendary-badge">⭐ Legendary</span>', unsafe_allow_html=True)
        st.markdown("")
        st.markdown('<span class="normal-badge">🔵 Non-Legendary</span>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET PREVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📂 Dataset Preview":
    st.markdown('<div class="section-header">📂 DATASET PREVIEW</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload your Pokémon CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state["df"] = df
        st.success(f"✅ Loaded {len(df)} records with {len(df.columns)} columns.")
    elif st.session_state["df"] is not None:
        df = st.session_state["df"]
        st.info("Using previously loaded dataset.")
    else:
        st.warning("Please upload your CSV file to continue.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric("Target Column", "Class" if "Class" in df.columns else "—")

    st.markdown("**First 10 Records:**")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("**Column Info & Data Types:**")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.values,
        "Non-Null Count": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values,
        "Sample": [df[c].iloc[0] for c in df.columns]
    })
    st.dataframe(info_df, use_container_width=True)

    st.markdown("**Class Distribution:**")
    target_col = None
    for c in ["Class", "class", "Legendary", "legendary", "Label", "label"]:
        if c in df.columns:
            target_col = c
            break
    if target_col:
        dist = df[target_col].value_counts().reset_index()
        dist.columns = ["Class", "Count"]
        dist["Percentage"] = (dist["Count"] / len(df) * 100).round(2).astype(str) + "%"
        st.dataframe(dist, use_container_width=True)

    st.markdown("**Descriptive Statistics (Numerical Columns):**")
    st.dataframe(df.describe().round(2), use_container_width=True)

    # Missing value check
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.warning(f"⚠️ Found {missing.sum()} missing values. They will be dropped during preprocessing.")
    else:
        st.success("✅ No missing values found!")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL SETUP & TRAIN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Model Setup & Train":
    st.markdown('<div class="section-header">⚙️ MODEL SETUP & TRAINING</div>', unsafe_allow_html=True)

    if st.session_state["df"] is None:
        st.warning("Please upload your dataset first (📂 Dataset Preview page).")
        st.stop()

    df = st.session_state["df"]

    # --- Feature & Target Selection ---
    st.subheader("1. Feature & Target Selection")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    default_features = [c for c in ["HP","Attack","Defense","Sp_Atk","Sp_Def","Speed"] if c in num_cols]
    feature_cols = st.multiselect(
        "Select numerical feature columns:", num_cols,
        default=default_features if default_features else num_cols[:6]
    )

    target_candidates = [c for c in cat_cols if c.lower() in ["class","legendary","label","target"]]
    target_col = st.selectbox(
        "Select target (class) column:",
        cat_cols,
        index=cat_cols.index(target_candidates[0]) if target_candidates else 0
    )

    if len(feature_cols) < 2:
        st.error("Select at least 2 feature columns.")
        st.stop()

    # --- Split & K ---
    st.subheader("2. Train-Test Split")
    split_ratio = st.select_slider(
        "Training set size:",
        options=[0.60, 0.65, 0.70, 0.75, 0.80],
        value=0.80,
        format_func=lambda x: f"{int(x*100)}% Train / {int((1-x)*100)}% Test"
    )
    random_seed = st.number_input("Random seed (for reproducibility):", value=42, step=1)

    st.subheader("3. Choose K Value")
    df_clean = df[feature_cols + [target_col]].dropna()
    X_all = df_clean[feature_cols]
    y_all = df_clean[target_col]
    X_tr_tmp, _, _, _ = train_test_split(X_all, y_all, train_size=split_ratio, random_state=int(random_seed))
    max_k = len(X_tr_tmp) - 1

    k_value = st.slider("K (number of nearest neighbors):", min_value=1, max_value=min(max_k, 25), value=5, step=2)
    st.caption(f"ℹ️ Odd K values (1, 3, 5, 7…) help avoid ties in binary classification. Max allowed: {max_k}")

    st.subheader("4. Feature Scaling Method")
    scaling = st.radio("Scaling method:", ["Min-Max Normalization", "No Scaling"])
    st.caption("⚠️ **KNN is distance-based.** Scaling is strongly recommended so no single feature dominates.")

    # --- Train button ---
    if st.button("🚀 Train the KNN Model"):
        # Clean & split
        df_clean = df[feature_cols + [target_col]].dropna()
        X = df_clean[feature_cols].values
        y = df_clean[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=split_ratio, random_state=int(random_seed), stratify=y
        )

        # Scale
        scaler = None
        if scaling == "Min-Max Normalization":
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        # Train sklearn KNN
        model = KNeighborsClassifier(n_neighbors=k_value, metric="euclidean")
        model.fit(X_train, y_train)

        # Store in session
        st.session_state.update({
            "df": df,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train.reset_index(drop=True),
            "y_test": y_test.reset_index(drop=True),
            "scaler": scaler,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "k_value": k_value,
            "model": model,
            "trained": True
        })

        st.success("✅ Model trained successfully!")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Training Records", len(X_train))
        col2.metric("Testing Records", len(X_test))
        col3.metric("K Value", k_value)
        col4.metric("Features", len(feature_cols))

        # Show workflow summary
        st.markdown("**KNN Workflow Applied:**")
        steps = [
            f"✅ Loaded {len(df_clean)} clean records",
            f"✅ Selected features: {', '.join(feature_cols)}",
            f"✅ Target: **{target_col}**",
            f"✅ Scaling: **{scaling}**",
            f"✅ Split: **{len(X_train)} train / {len(X_test)} test** (seed={int(random_seed)})",
            f"✅ K = **{k_value}** nearest neighbors",
            f"✅ Distance metric: **Euclidean**"
        ]
        for s in steps:
            st.markdown(s)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EVALUATION RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Evaluation Results":
    st.markdown('<div class="section-header">📊 EVALUATION RESULTS</div>', unsafe_allow_html=True)

    if not st.session_state["trained"]:
        st.warning("Train the model first (⚙️ Model Setup & Train page).")
        st.stop()

    model   = st.session_state["model"]
    X_test  = st.session_state["X_test"]
    y_test  = st.session_state["y_test"]
    k_value = st.session_state["k_value"]

    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="Legendary", zero_division=0)
    rec  = recall_score(y_test, y_pred, pos_label="Legendary", zero_division=0)
    f1   = f1_score(y_test, y_pred, pos_label="Legendary", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    # Metrics cards
    st.subheader("Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{acc:.1%}</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{prec:.1%}</div><div class="metric-label">Precision</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{rec:.1%}</div><div class="metric-label">Recall</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-value">{f1:.1%}</div><div class="metric-label">F1-Score</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        labels = sorted(y_test.unique())
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr",
                    xticklabels=labels, yticklabels=labels,
                    linewidths=2, linecolor="white", ax=ax,
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
        ax.set_ylabel("Actual Label", fontsize=12, fontweight="bold")
        ax.set_title(f"Confusion Matrix (K={k_value})", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Interpretation
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        st.markdown(f"""
        **Matrix Interpretation:**
        - ✅ **True Positives (TP):** {tp} — Correctly predicted Legendary
        - ✅ **True Negatives (TN):** {tn} — Correctly predicted Non-Legendary
        - ❌ **False Positives (FP):** {fp} — Non-Legendary predicted as Legendary
        - ❌ **False Negatives (FN):** {fn} — Legendary missed (predicted Non-Legendary)
        """)

    with col2:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Metric Formulas Used")
        st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
        st.latex(r"\text{Precision} = \frac{TP}{TP + FP}")
        st.latex(r"\text{Recall} = \frac{TP}{TP + FN}")
        st.latex(r"\text{F1} = 2 \cdot \frac{Precision \times Recall}{Precision + Recall}")

    # Test predictions table
    st.subheader("Test Set Predictions (First 20)")
    pred_df = pd.DataFrame({
        "Actual": y_test.values[:20],
        "Predicted": y_pred[:20],
        "Correct?": ["✅" if a == p else "❌" for a, p in zip(y_test.values[:20], y_pred[:20])]
    })
    st.dataframe(pred_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: NEW RECORD PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 New Record Prediction":
    st.markdown('<div class="section-header">🔮 CLASSIFY A NEW POKÉMON</div>', unsafe_allow_html=True)

    if not st.session_state["trained"]:
        st.warning("Train the model first (⚙️ Model Setup & Train page).")
        st.stop()

    df           = st.session_state["df"]
    feature_cols = st.session_state["feature_cols"]
    target_col   = st.session_state["target_col"]
    scaler       = st.session_state["scaler"]
    X_train      = st.session_state["X_train"]
    y_train      = st.session_state["y_train"]
    k_value      = st.session_state["k_value"]

    st.markdown("Enter the base stats for the Pokémon you want to classify:")

    # Compute column stats for slider ranges
    col_stats = df[feature_cols].describe()

    inputs = {}
    cols = st.columns(min(len(feature_cols), 3))
    for i, feat in enumerate(feature_cols):
        col = cols[i % len(cols)]
        min_v = float(col_stats.loc["min", feat])
        max_v = float(col_stats.loc["max", feat])
        mean_v = float(col_stats.loc["mean", feat])
        inputs[feat] = col.number_input(
            f"{feat}", min_value=0.0, max_value=max_v * 1.5,
            value=round(mean_v, 1), step=1.0
        )

    pokemon_name = st.text_input("Pokémon Name (optional, for display):", placeholder="e.g. MissingNo")

    if st.button("⚡ Classify This Pokémon!"):
        query_raw = np.array([inputs[f] for f in feature_cols])

        # Scale if needed
        if scaler is not None:
            query_scaled = scaler.transform(query_raw.reshape(1, -1))[0]
        else:
            query_scaled = query_raw

        predicted, neighbors = knn_predict_with_neighbors(
            X_train, y_train, query_scaled, k_value
        )

        # Result display
        st.markdown("---")
        name_display = f"**{pokemon_name}**" if pokemon_name else "This Pokémon"
        if predicted == "Legendary":
            st.markdown(f'<div style="text-align:center; padding:1.5rem; background:linear-gradient(135deg,#FFCB05,#FF6B00); border-radius:16px;">'
                        f'<div style="font-size:1.8rem; font-weight:900; color:#1a1a2e;">⭐ LEGENDARY ⭐</div>'
                        f'<div style="color:#1a1a2e; font-size:1rem;">{name_display} is predicted to be <b>Legendary</b>!</div>'
                        f'</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="text-align:center; padding:1.5rem; background:linear-gradient(135deg,#2A75BB,#1a5c9e); border-radius:16px;">'
                        f'<div style="font-size:1.8rem; font-weight:900; color:white;">🔵 NON-LEGENDARY</div>'
                        f'<div style="color:#ddd; font-size:1rem;">{name_display} is predicted to be <b>Non-Legendary</b>.</div>'
                        f'</div>', unsafe_allow_html=True)

        # Nearest neighbors table
        st.markdown(f"### The {k_value} Nearest Neighbors")
        legend_count = sum(1 for n in neighbors if n["class"] == "Legendary")
        nonlegend_count = k_value - legend_count

        st.markdown(f"**Vote count:** ⭐ Legendary: {legend_count} | 🔵 Non-Legendary: {nonlegend_count}")

        df_clean = df[feature_cols + [target_col]].dropna().reset_index(drop=True)
        for nb in neighbors:
            cls = nb["class"]
            badge = "⭐ Legendary" if cls == "Legendary" else "🔵 Non-Legendary"
            row = df_clean.iloc[nb["index"]]
            name_val = df.iloc[nb["index"]]["Name"] if "Name" in df.columns else f"Record #{nb['index']}"
            stats_str = " | ".join([f"{f}: {row[f]}" for f in feature_cols])
            st.markdown(
                f'<div class="neighbor-card">'
                f'<b>#{nb["rank"]}</b> — <b>{name_val}</b> &nbsp; <span style="background:#eee;padding:2px 8px;border-radius:12px;">{badge}</span><br>'
                f'<span style="color:#666;font-size:0.85rem;">{stats_str}</span><br>'
                f'<span style="color:#2A75BB;font-size:0.85rem;">Distance: {nb["distance"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Distance bar chart
        st.subheader("Distance Ranking Chart")
        fig, ax = plt.subplots(figsize=(7, 3))
        colors = ["#FFCB05" if n["class"] == "Legendary" else "#2A75BB" for n in neighbors]
        bars = ax.barh([f"#{n['rank']}" for n in neighbors],
                       [n["distance"] for n in neighbors],
                       color=colors, edgecolor="white", linewidth=1.5)
        ax.set_xlabel("Euclidean Distance", fontweight="bold")
        ax.set_title(f"Top {k_value} Nearest Neighbors by Distance", fontweight="bold")
        legend_patch = mpatches.Patch(color='#FFCB05', label='Legendary')
        normal_patch = mpatches.Patch(color='#2A75BB', label='Non-Legendary')
        ax.legend(handles=[legend_patch, normal_patch], loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Explanation
        majority = "Legendary" if legend_count > nonlegend_count else "Non-Legendary"
        st.markdown(f"""
        <div class="explanation-box">
        <b>🧠 Why this prediction was made:</b><br><br>
        The system computed the <b>Euclidean distance</b> between the new Pokémon's stats
        ({", ".join([f"{f}={inputs[f]}" for f in feature_cols])})
        and every Pokémon in the training set.<br><br>
        The <b>{k_value} closest Pokémon</b> were selected as "neighbors."
        Of those, <b>{legend_count}</b> are Legendary and <b>{nonlegend_count}</b> are Non-Legendary.<br><br>
        By <b>majority vote</b>, the prediction is: <b>{majority}</b>.
        The nearest neighbor had a distance of <b>{neighbors[0]["distance"]}</b>.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Visualizations":
    st.markdown('<div class="section-header">📈 VISUALIZATIONS</div>', unsafe_allow_html=True)

    if st.session_state["df"] is None:
        st.warning("Please upload your dataset first.")
        st.stop()

    df = st.session_state["df"]
    feature_cols = st.session_state.get("feature_cols") or [c for c in ["HP","Attack","Defense","Sp_Atk","Sp_Def","Speed"] if c in df.columns]
    target_col   = st.session_state.get("target_col") or next((c for c in ["Class","class"] if c in df.columns), None)

    if not target_col:
        st.error("Target column not identified. Train the model first.")
        st.stop()

    colors_map = {"Legendary": "#FFCB05", "Non-Legendary": "#2A75BB"}

    # Chart 1: Class Distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Class Distribution")
        dist = df[target_col].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(dist.index, dist.values,
                      color=[colors_map.get(c, "#aaa") for c in dist.index],
                      edgecolor="white", linewidth=2)
        ax.bar_label(bars, fmt="%d", padding=3, fontweight="bold")
        ax.set_ylabel("Count", fontweight="bold")
        ax.set_title("Legendary vs Non-Legendary", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Chart 2: Feature Boxplots
    with col2:
        st.subheader("2. Feature Distribution by Class")
        if feature_cols:
            feat_choice = st.selectbox("Choose feature to inspect:", feature_cols)
            fig, ax = plt.subplots(figsize=(5, 4))
            groups = [df[df[target_col] == cls][feat_choice].dropna().values
                      for cls in ["Legendary", "Non-Legendary"] if cls in df[target_col].values]
            group_labels = [cls for cls in ["Legendary", "Non-Legendary"] if cls in df[target_col].values]
            bp = ax.boxplot(groups, labels=group_labels, patch_artist=True,
                            medianprops=dict(color="black", linewidth=2))
            for patch, label in zip(bp["boxes"], group_labels):
                patch.set_facecolor(colors_map.get(label, "#aaa"))
                patch.set_alpha(0.7)
            ax.set_ylabel(feat_choice, fontweight="bold")
            ax.set_title(f"{feat_choice} by Class", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Chart 3: Scatterplot of 2 features
    st.subheader("3. Scatterplot: Two Features by Class")
    c1, c2 = st.columns(2)
    fx = c1.selectbox("X-axis feature:", feature_cols, index=0)
    fy = c2.selectbox("Y-axis feature:", feature_cols, index=min(1, len(feature_cols)-1))
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in colors_map.items():
        subset = df[df[target_col] == label]
        if len(subset):
            ax.scatter(subset[fx], subset[fy],
                       c=color, label=label, alpha=0.7,
                       edgecolors="white", linewidth=0.5, s=60)
    ax.set_xlabel(fx, fontweight="bold")
    ax.set_ylabel(fy, fontweight="bold")
    ax.set_title(f"{fx} vs {fy} — Colored by Class", fontweight="bold")
    ax.legend(title="Class")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Chart 4: Radar chart of mean stats
    st.subheader("4. Average Base Stats Radar Chart")
    if feature_cols:
        legendary_means    = df[df[target_col] == "Legendary"][feature_cols].mean().values
        nonlegendary_means = df[df[target_col] == "Non-Legendary"][feature_cols].mean().values

        N = len(feature_cols)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        for values, color, label in [
            (legendary_means, "#FFCB05", "Legendary"),
            (nonlegendary_means, "#2A75BB", "Non-Legendary")
        ]:
            vals = values.tolist() + values[:1].tolist()
            ax.plot(angles, vals, color=color, linewidth=2, linestyle='solid')
            ax.fill(angles, vals, color=color, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), feature_cols)
        ax.set_title("Average Stats: Legendary vs Non-Legendary", fontweight="bold", pad=20)
        legend_patch = mpatches.Patch(color='#FFCB05', label='Legendary', alpha=0.7)
        normal_patch = mpatches.Patch(color='#2A75BB', label='Non-Legendary', alpha=0.7)
        ax.legend(handles=[legend_patch, normal_patch], loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
