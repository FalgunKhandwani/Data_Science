import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report, silhouette_score, \
    confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import warnings
import matplotlib.pyplot as plt # Import matplotlib for decision tree plotting

# --- Page Config ---
st.set_page_config(
    page_title="Fire Response ML Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Define Constants ---
# FIX: Defined month_columns globally so it can be accessed by all parts of the app.
MONTH_COLUMNS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


# --- Data Loading (Cached) ---
@st.cache_data(show_spinner="Loading and preparing data...")
def load_data():
    """
    Loads, cleans, and prepares the fire response data.
    """
    try:
        # Load the dataset
        # FIX: Added header=0 and encoding='utf-8-sig' for robustness.
        # This explicitly tells pandas to use the first row as the header
        # and correctly handles potential invisible characters (BOM)
        # that can be added by Windows editors and cause a KeyError.
        df = pd.read_csv("2019_Fire_Response_Time_Analysis.csv", header=0, encoding='utf-8-sig')
    except FileNotFoundError:
        st.error("Error: `2019_Fire_Response_Time_Analysis.csv` not found.")
        st.info("Please add the file to the same directory as `app.py` and refresh.")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None, None, None, None, None, None

    # --- 1. Data Cleaning ---

    # Clean column names: remove leading/trailing spaces
    df.columns = df.columns.str.strip()

    # Drop the _id column as it's just an index
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])

    # --- 2. Handle Missing Data ---
    # Drop rows where the target 'Response Times' is missing (this drops the first empty data row)
    if 'Response Times' not in df.columns:
        st.error("Data loading error: 'Response Times' column not found in the CSV.")
        st.info("Please ensure your CSV file has the correct headers (e.g., 'Response Times', 'YTD', 'Jan', 'Feb', etc.).")
        return None, None, None, None, None, None

    df = df.dropna(subset=['Response Times'])

    # Check if all month columns exist
    missing_months = [col for col in MONTH_COLUMNS if col not in df.columns]
    if missing_months:
        st.error(f"Data loading error: The following month columns are missing: {', '.join(missing_months)}")
        return None, None, None, None, None, None

    # Fill NaNs in month columns with 0 (assuming 0 incidents)
    for col in MONTH_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Also fill YTD
    if 'YTD' not in df.columns:
        st.error("Data loading error: 'YTD' column not found.")
        return None, None, None, None, None, None

    df['YTD'] = pd.to_numeric(df['YTD'], errors='coerce').fillna(0)

    # --- 3. Feature Engineering ---
    # Create target (y) for classification
    y_class = df['Response Times']

    # Create features (X) for classification (all month data)
    X_class = df[MONTH_COLUMNS]

    # --- 4. Prepare Data for Regression ---
    # Target for regression (YTD)
    y_reg = df['YTD']

    # Features for Simple Linear Regression (e.g., predict YTD from Jan)
    X_reg_simple = df[['Jan']]

    # Features for Multiple Linear Regression (e.g., predict YTD from all months)
    X_reg_multi = df[MONTH_COLUMNS]

    # Simple sanity check
    if df.empty or X_class.empty or y_class.empty:
        st.error("Data is empty after processing. Please check the CSV file.")
        return None, None, None, None, None, None

    return df, X_class, y_class, y_reg, X_reg_simple, X_reg_multi


# --- Main App Execution ---
st.title("🔥 Fire Response Time ML Dashboard")
st.markdown("""
This dashboard analyzes the 2019 Fire Response Time dataset.
Use the sidebar to navigate between pages for Exploratory Data Analysis (EDA),
Model Performance, and Predictions.
""")

# Load data
df, X_class, y_class, y_reg, X_reg_simple, X_reg_multi = load_data()

# Stop execution if data loading failed
if df is None:
    st.stop()

# --- Preprocessing & Model Caching ---
# We need to split the data *before* caching models
# We'll use the full dataset for prediction, but splits for evaluation
# FIX: Removed stratify=y_class because the dataset is too small and has
# classes with only 1 member, which causes train_test_split to fail.
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(X_class, y_class, test_size=0.3,
                                                                            random_state=42)

X_reg_multi_train, X_reg_multi_test, y_reg_train, y_reg_test = train_test_split(X_reg_multi, y_reg, test_size=0.3,
                                                                                random_state=42)

X_reg_simple_train, X_reg_simple_test, y_reg_simple_train, y_reg_simple_test = train_test_split(X_reg_simple, y_reg,
                                                                                                test_size=0.3,
                                                                                                random_state=42)

# Define scalers
scaler_class = StandardScaler()
scaler_reg = StandardScaler()

# Fit scalers on training data
X_class_train_scaled = scaler_class.fit_transform(X_class_train)
# FIX: Corrected typo from X_C_class_test to X_class_test
X_class_test_scaled = scaler_class.transform(X_class_test)

X_reg_multi_train_scaled = scaler_reg.fit_transform(X_reg_multi_train)
X_reg_multi_test_scaled = scaler_reg.transform(X_reg_multi_test)

X_reg_simple_train_scaled = scaler_reg.fit_transform(X_reg_simple_train)
X_reg_simple_test_scaled = scaler_reg.transform(X_reg_simple_test)


# --- Helper Function for Model Features ---
def show_model_features(features, explanation):
    """Displays the features used by a model."""
    with st.expander("**Features Used & Explanation**"):
        st.markdown(f"""
        This model was trained using the following column(s) as features:
        - **`{', '.join(features)}`**

        **Why?**
        {explanation}
        """)


# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Exploratory Data Analysis (EDA)",
    "📈 Model Performance",
    "🔮 Prediction Studio"
])

# ==============================================================================
# --- Page 1: Exploratory Data Analysis (EDA) ---
# ==============================================================================
if page == "📊 Exploratory Data Analysis (EDA)":
    st.header("📊 Exploratory Data Analysis")
    st.markdown("This section provides a visual overview of the dataset, based on the EDA notebook.")

    with st.container(border=True):
        st.subheader("Raw Data Table (Cleaned)", divider="rainbow")
        st.markdown(f"The loaded dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
        st.dataframe(df)

    with st.container(border=True):
        st.subheader("1. Total Incidents per Response Time (YTD)", divider="rainbow")
        st.markdown(
            "This chart shows the total number of incidents (Year-to-Date) for each response time category. "
            "It highlights which response times are most common.")
        fig1 = px.bar(
            df,
            x='Response Times',
            y='YTD',
            title='Total Incidents (YTD) by Response Time Category',
            color='Response Times',
            labels={'YTD': 'Total Incidents (Year-to-Date)'}
        )
        fig1.update_layout(xaxis_title="Response Time Category", yaxis_title="Total Incidents")
        st.plotly_chart(fig1, use_container_width=True)

    with st.container(border=True):
        st.subheader("2. Monthly Incident Trends", divider="rainbow")
        st.markdown("This line chart shows the total number of incidents across all categories for each month, "
                    "revealing seasonal patterns in fire incidents.")
        monthly_totals = df[MONTH_COLUMNS].sum().reset_index()
        monthly_totals.columns = ['Month', 'Total Incidents']

        fig2 = px.line(
            monthly_totals,
            x='Month',
            y='Total Incidents',
            title='Total Fire Incidents per Month',
            markers=True,
            labels={'Total Incidents': 'Number of Incidents'}
        )
        fig2.update_layout(xaxis_title="Month", yaxis_title="Total Incidents")
        st.plotly_chart(fig2, use_container_width=True)

    with st.container(border=True):
        st.subheader("3. Heatmap of Incidents", divider="rainbow")
        st.markdown("This heatmap visualizes the number of incidents for each response time *and* month. "
                    "Darker squares indicate a higher volume of incidents.")
        df_heatmap = df.set_index('Response Times')[MONTH_COLUMNS]
        fig3 = px.imshow(
            df_heatmap,
            title='Heatmap of Incidents by Response Time and Month',
            color_continuous_scale='Reds',
            aspect="auto"
        )
        fig3.update_layout(xaxis_title="Month", yaxis_title="Response Time Category")
        st.plotly_chart(fig3, use_container_width=True)

# ==============================================================================
# --- Page 2: Model Performance ---
# ==============================================================================
elif page == "📈 Model Performance":
    st.header("📈 Model Performance Evaluation")
    st.markdown("This page evaluates the different models you requested. Use the tabs to explore each model type.")

    tab1, tab2, tab3 = st.tabs(["Regression Models", "Classification Models", "Clustering Models"])

    # --- Regression Tab ---
    with tab1:
        st.subheader("Regression: Predicting Total Incidents (YTD)")
        st.markdown(
            "Regression models predict a continuous numerical value. Here, we attempt to predict the `YTD` total.")
        st.info(
            "**Note:** Predicting `YTD` from the months is a flawed model, as `YTD` is just the sum of the months. "
            "A perfect model would have an R² of 1.0. This is purely for demonstrating the algorithms.", icon="ℹ️")

        reg_model_choice = st.selectbox(
            "Select a Regression Model:",
            ["Simple Linear Regression", "Multiple Linear Regression"],
            key="reg_select"
        )

        # --- Simple Linear Regression ---
        if reg_model_choice == "Simple Linear Regression":
            st.markdown("This model predicts the `YTD` total using *only* the `Jan` column.")
            show_model_features(
                features=['Jan'],
                explanation="We use one feature (`Jan`) to predict one target (`YTD`). This helps visualize a simple 2D relationship."
            )

            # Train model
            model = LinearRegression()
            model.fit(X_reg_simple_train_scaled, y_reg_simple_train)
            y_pred = model.predict(X_reg_simple_test_scaled)

            # Metrics
            r2 = r2_score(y_reg_simple_test, y_pred)
            mae = mean_absolute_error(y_reg_simple_test, y_pred)

            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            col1.metric("R-squared (R²)", f"{r2:.3f}")
            col2.metric("Mean Absolute Error (MAE)", f"{mae:.2f} incidents")
            st.markdown(
                f"An R² of **{r2:.3f}** means the model explains **{r2 * 100:.1f}%** of the variance in `YTD` using *only* `Jan` data.")

            st.subheader("Visual: Actual vs. Predicted")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X_reg_simple_test['Jan'], y=y_reg_simple_test, mode='markers', name='Actual Data',
                                     marker=dict(color='blue')))
            fig.add_trace(go.Scatter(x=X_reg_simple_test['Jan'], y=y_pred, mode='lines', name='Regression Line',
                                     line=dict(color='red', dash='dash')))
            fig.update_layout(title="Simple Linear Regression (YTD vs. Jan)",
                              xaxis_title="Incidents in Jan", yaxis_title="YTD Incidents")
            st.plotly_chart(fig, use_container_width=True)

        # --- Multiple Linear Regression ---
        elif reg_model_choice == "Multiple Linear Regression":
            st.markdown("This model predicts the `YTD` total using *all 12 month columns*.")
            show_model_features(
                features=MONTH_COLUMNS,
                explanation="We use all 12 features (months) to predict one target (`YTD`). As expected, this model is "
                            "nearly perfect because the features add up to the target."
            )

            # Train model
            model = LinearRegression()
            model.fit(X_reg_multi_train_scaled, y_reg_train)
            y_pred = model.predict(X_reg_multi_test_scaled)

            # Metrics
            r2 = r2_score(y_reg_test, y_pred)
            mae = mean_absolute_error(y_reg_test, y_pred)

            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            col1.metric("R-squared (R²)", f"{r2:.3f}")
            col2.metric("Mean Absolute Error (MAE)", f"{mae:.2f} incidents")
            st.markdown(
                f"An R² of **{r2:.3f}** is (correctly) near-perfect, as the 12 months sum up to the `YTD` total.")

            st.subheader("Visual: Actual vs. Predicted")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=y_reg_test, y=y_pred, mode='markers', name='Actual vs. Predicted', marker=dict(color='blue')))
            fig.add_trace(go.Scatter(x=[y_reg_test.min(), y_reg_test.max()], y=[y_reg_test.min(), y_reg_test.max()],
                                     mode='lines', name='Ideal Fit', line=dict(color='red', dash='dash')))
            fig.update_layout(title="Multiple Linear Regression (Actual vs. Predicted YTD)",
                              xaxis_title="Actual YTD", yaxis_title="Predicted YTD")
            st.plotly_chart(fig, use_container_width=True)

    # --- Classification Tab ---
    with tab2:
        st.subheader("Classification: Predicting Response Time Category")
        st.markdown(
            "Classification models predict a specific category (or 'class'). "
            "This is the most useful model: **predicting the `Response Times` category based on the 12 months of incident data.**")

        class_model_choice = st.selectbox(
            "Select a Classification Model:",
            ["Logistic Regression", "Decision Tree"],
            key="class_select"
        )

        show_model_features(
            features=MONTH_COLUMNS,
            explanation="We use the 12 month columns to predict the *category* of `Response Times`."
        )

        model = None
        model_name = ""

        # --- Logistic Regression ---
        if class_model_choice == "Logistic Regression":
            model_name = "Logistic Regression"
            with st.spinner(f"Training {model_name}..."):
                model = LogisticRegression(max_iter=1000, random_state=42)
                model.fit(X_class_train_scaled, y_class_train)

        # --- Decision Tree ---
        elif class_model_choice == "Decision Tree":
            model_name = "Decision Tree"
            with st.spinner(f"Training {model_name}..."):
                model = DecisionTreeClassifier(max_depth=5, random_state=42)
                model.fit(X_class_train_scaled, y_class_train)

        # --- Evaluate Classification Model ---
        if model:
            y_pred = model.predict(X_class_test_scaled)
            acc = accuracy_score(y_class_test, y_pred)

            st.subheader(f"{model_name} Performance")
            st.metric("Accuracy Score", f"{acc * 100:.1f}%")
            st.markdown(
                f"This model correctly predicted the `Response Times` category for **{acc * 100:.1f}%** of the test data.")

            st.subheader("Visual: Confusion Matrix")
            st.markdown(
                "This matrix shows what the model predicted (columns) vs. what was true (rows). "
                "The diagonal shows correct predictions.")
            
            # FIX: Create a combined list of all labels present in EITHER the test set or the predictions.
            # This handles the case where the random split removes a class from the test set,
            # which would crash confusion_matrix() if we used labels=model.classes_.
            all_cm_labels = sorted(np.unique(np.concatenate((y_class_test, y_pred))))
            
            cm = confusion_matrix(y_class_test, y_pred, labels=all_cm_labels)
            fig = px.imshow(
                cm,
                text_auto=True,
                labels=dict(x="Predicted", y="True"),
                x=all_cm_labels,
                y=all_cm_labels,
                color_continuous_scale='Blues'
            )
            fig.update_layout(title=f"Confusion Matrix for {model_name}")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Classification Report")
            st.text("This report shows the precision, recall, and f1-score for each class.")
            # Use warnings.catch_warnings to suppress UndefinedMetricWarning for small datasets
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # FIX: Use the same all_cm_labels for the report to ensure consistency.
                report = classification_report(y_class_test, y_pred, labels=all_cm_labels, zero_division=0)
                st.code(report, language='text')

            # Show Decision Tree plot
            if model_name == "Decision Tree":
                with st.expander("Show Decision Tree Visualization"):
                    fig, ax = plt.subplots(figsize=(20, 10))
                    plot_tree(model,
                              feature_names=MONTH_COLUMNS,
                              class_names=model.classes_, # model.classes_ is fine here, it's just for labeling the nodes
                              filled=True,
                              rounded=True,
                              ax=ax,
                              fontsize=10)
                    st.pyplot(fig)
        else:
            st.info("Select a model to train and evaluate.")

    # --- Clustering Tab ---
    with tab3:
        st.subheader("Clustering: Grouping Response Time Categories")
        st.markdown(
            "Clustering is an *unsupervised* model that finds patterns. "
            "Here, we use **K-Means** to group the `Response Times` categories based on their 12-month incident patterns.")

        show_model_features(
            features=MONTH_COLUMNS,
            explanation="The model groups the rows (`Response Times` categories) into clusters based on their "
                        "similarities across the 12 month columns."
        )

        # Scale all classification data for clustering
        X_scaled = scaler_class.fit_transform(X_class)

        try:
            k = st.slider("Select Number of Clusters (k)", min_value=2, max_value=6, value=3)

            with st.spinner(f"Running K-Means with k={k}..."):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                # FIX: Corrected typo from X_S_scaled to X_scaled
                clusters = kmeans.fit_predict(X_scaled)

                # Add cluster data back to original df
                df_cluster = df.copy()
                df_cluster['Cluster'] = clusters.astype(str)

                # Performance
                st.subheader("Model Performance")
                score = silhouette_score(X_scaled, clusters)
                st.metric("Silhouette Score", f"{score:.3f}")
                st.markdown(
                    "A score closer to 1 indicates well-defined clusters. "
                    "A score near 0 indicates overlapping clusters.")

                # Visuals
                st.subheader("Visual: Cluster Profiles (Heatmap)")
                st.markdown("This heatmap shows the average monthly pattern for each cluster.")
                cluster_profile = df_cluster.groupby('Cluster')[MONTH_COLUMNS].mean()
                fig_hm = px.imshow(
                    cluster_profile,
                    title=f'Average Monthly Incidents per Cluster (k={k})',
                    color_continuous_scale='Viridis',
                    aspect="auto"
                )
                st.plotly_chart(fig_hm, use_container_width=True)

                st.subheader("Visual: Cluster Scatter Plot")
                st.markdown(
                    "This 2D scatter plot (using Jan vs. Feb for simplicity) shows how the clusters are separated.")
                fig_scatter = px.scatter(
                    df_cluster,
                    x='Jan',
                    y='Feb',
                    color='Cluster',
                    hover_name='Response Times',
                    title=f'Clusters (Jan vs. Feb)',
                    color_discrete_map={str(i): px.colors.qualitative.Plotly[i] for i in range(k)}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during clustering. This can happen if k is larger than the number of data points.")
            st.error(f"Details: {e}")

# ==============================================================================
# --- Page 3: Prediction Studio ---
# ==============================================================================
elif page == "🔮 Prediction Studio":
    st.header("🔮 Prediction Studio")
    st.markdown("Use a trained model to make a prediction on new, hypothetical data.")
    st.info(
        "**This page uses the Classification model (Logistic Regression) as it's the most useful.** "
        "Enter the number of incidents you *expect* for each month to predict the `Response Times` category.", icon="ℹ️")

    # Train the final prediction model on ALL data
    @st.cache_resource
    def get_prediction_model():
        model = LogisticRegression(max_iter=1000, random_state=42)
        # We need a scaler fitted on ALL data for the prediction model
        scaler = StandardScaler().fit(X_class)
        model.fit(scaler.transform(X_class), y_class)
        return model, scaler

    pred_model, pred_scaler = get_prediction_model()

    with st.container(border=True):
        st.subheader("Enter Monthly Incident Data:")
        input_data = {}

        # Create 3 columns for input
        cols = st.columns(3)
        col_idx = 0
        for month in MONTH_COLUMNS:
            with cols[col_idx % 3]:
                # Use mean of data as default
                default_val = float(X_class[month].mean())
                input_data[month] = st.number_input(f"{month} Incidents", min_value=0, value=int(default_val), step=10)
            col_idx += 1

        if st.button("Predict Response Time Category", type="primary", use_container_width=True):
            # Create a DataFrame for the single prediction
            input_df = pd.DataFrame([input_data])
            # Ensure column order is correct
            input_df = input_df[MONTH_COLUMNS]

            # Scale the input data
            input_scaled = pred_scaler.transform(input_df)

            # Make prediction
            prediction = pred_model.predict(input_scaled)
            prediction_proba = pred_model.predict_proba(input_scaled)

            st.success(f"**Predicted Response Time Category:**")
            st.metric(
                label="Prediction",
                value=f"{prediction[0]}"
            )

            with st.expander("Show Prediction Probabilities"):
                st.markdown("This shows the model's 'confidence' for each possible category.")
                proba_df = pd.DataFrame(prediction_proba, columns=pred_model.classes_)
                st.dataframe(proba_df.T.rename(columns={0: 'Probability'}).style.format("{:.1%}"))

