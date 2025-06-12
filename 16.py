import streamlit as st
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.multiclass import type_of_target
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import zscore
import base64
from fpdf import FPDF
from scipy.stats import skew, kurtosis
from pandas.api.types import is_numeric_dtype
import os, psutil
import streamlit as st
import pandas as pd
import plotly.express as px
from flaml import AutoML
from pprint import pprint
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.markdown("""
    <style>
        /* Overall page style */
        .main {
            background-color: #f5f7fa;
            padding: 2rem;
            border-radius: 12px;
            margin-top: 20px;
        }

        h1 {
            color: #1f77b4;
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
            margin-bottom: 10px;
        }

        /* Button styling */
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 0.5rem 1.5rem;
            font-size: 16px;
        }

        .stButton>button:hover {
            background-color: #125d98;
        }

        /* Subheader style */
        .stMarkdown h2, .stMarkdown h3 {
            color: #333333;
            font-family: 'Segoe UI', sans-serif;
        }

        /* File uploader */
        .stFileUploader label {
            color: #2c3e50;
            font-weight: bold;
        }

        /* Selectbox and sliders */
        .stSelectbox label, .stSlider label {
            color: #2c3e50;
            font-weight: 600;
        }

        /* Dataframe table */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
        }

        /* Warning and success messages */
        .stAlert {
            border-radius: 10px !important;
        }

        /* Markdown paragraphs */
        p {
            font-family: 'Segoe UI', sans-serif;
            font-size: 16px;
        }

    </style>
""", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv","xlsx", "xls"])
if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1]
    if file_type == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_type in ['xlsx', 'xls']:
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file format.")
        st.stop()
    st.subheader("Raw Data")
    with st.expander("Preview Data"):
        st.dataframe(df)
    with st.expander("Basic Data Insights"):
        st.subheader("Dataset Insights")
        process = psutil.Process(os.getpid())
        st.write(f"Memory used: {process.memory_info().rss / (1024 * 1024):.2f} MB")
        st.markdown(f"- **Number of rows:** {df.shape[0]}")
        st.markdown(f"- **Number of columns:** {df.shape[1]}")
        st.subheader("Column Data Types")
        st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]))
        st.subheader("Descriptive Statistics (Numerical Columns)")
        st.dataframe(df.describe().T)
        st.subheader("Unique Values per Column")
        unique_vals = df.nunique().sort_values(ascending=False)
        st.dataframe(unique_vals)
    with st.expander("Categorization Of Data"):
        st.subheader("Most Frequent Values (Top Categories)")
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].nunique() < 20:
                st.write(f"🔹 **{col}** - Top Values:")
                st.dataframe(df[col].value_counts().head(5))
        st.subheader("Skewness of Numerical Features")
        skewness = df.select_dtypes(include='number').skew().sort_values(ascending=False)
        st.dataframe(skewness)
        st.subheader("Zero Count Per Column")
        zero_counts = (df == 0).sum()
        st.dataframe(zero_counts[zero_counts > 0])
        st.subheader("Duplicate Rows")
        duplicate_count = df.duplicated().sum()
        st.write(f"Duplicate rows in dataset: **{duplicate_count}**")
    from sklearn.impute import SimpleImputer
    with st.expander(" Missing Values (Must do for further operations)"):
        st.subheader(" Missing Values Detection")
        missing = df.isnull().sum()
        total_missing = missing[missing > 0]
        st.write(" **Total missing values per column:**")
        st.dataframe(total_missing)
        st.subheader(" Handle Missing Values")
        option = st.selectbox(
            "Select a strategy to handle missing values:",
            [
                "Select an option",
                "Remove rows with missing values",
                "Remove columns with missing values",
                "Impute missing values (Mean)"
            ]
            )
        if "df_cleaned" not in st.session_state:
            st.session_state.df_cleaned = df.copy()
        if option == "Remove rows with missing values":
                df_temp = df.dropna()
                if df_temp.shape[0] == 0:
                    st.error(" All rows removed. Dataset is empty! Choose a different option.")
                else:
                    st.success(f" Removed rows. {df_temp.shape[0]} rows remain.")
                    st.session_state.df_cleaned = df_temp
                    st.dataframe(df_temp)
        elif option == "Remove columns with missing values":
                df_temp = df.dropna(axis=1)
                if df_temp.shape[1] == 0:
                    st.error(" All columns removed. Dataset is empty! Choose a different option.")
                else:
                    st.success(f" Removed columns. {df_temp.shape[1]} columns remain.")
                    st.session_state.df_cleaned = df_temp
                    st.dataframe(df_temp)
        elif option == "Impute missing values (Mean)":
                try:
                    df_numeric = df.select_dtypes(include=['float64', 'int64'])
                    df_non_numeric = df.select_dtypes(exclude=['float64', 'int64'])
                    imputer = SimpleImputer(strategy='mean')
                    imputed_numeric = imputer.fit_transform(df_numeric)
                    if imputed_numeric.shape[0] == 0:
                        st.error(" Imputation failed: No rows remain after imputation.")
                    else:
                        df_imputed = pd.DataFrame(imputed_numeric, columns=df_numeric.columns)
                        df_temp = pd.concat([df_imputed, df_non_numeric.reset_index(drop=True)], axis=1)
                        st.session_state.df_cleaned = df_temp
                        st.success(" Missing values imputed with mean.")
                        st.dataframe(df_temp)
                except Exception as e:
                    st.error(f"Imputation failed: {e}")
        elif option == "Select an option":
                st.info(" Please select how you'd like to handle missing values.")
    # Final DataFrame for all downstream use
    df = st.session_state.df_cleaned
    # Extra safeguard to prevent passing empty df to ML
    if df.shape[0] == 0 or df.shape[1] == 0:
        st.stop()  # 🚨 Immediately stops app execution if df is invalid
    st.subheader("Updated Data")
    st.dataframe(st.session_state.df_cleaned)
    with st.expander("Target Column Distribution"):
    # Target column selection
        target_column = st.selectbox("Select Target Column (Label)", df.columns)
        if target_column:
            if df[target_column].dtype == 'object' or df[target_column].nunique() < 20:
                st.subheader(f"Barchart for Target Column: {target_column}")
                vc = df[target_column].astype(str).value_counts().reset_index()
                vc.columns = ['Category', 'Count']
                vc['Category'] = vc['Category'].astype(str)
                st.bar_chart(vc.set_index('Category'))
            else:
                st.subheader(f"Boxplot for Target Column: {target_column}")
                fig, ax = plt.subplots()
                sns.boxplot(x=df[target_column], ax=ax)
                ax.set_title(f"Boxplot of {target_column}")
                st.pyplot(fig)
    if "clean_df" not in st.session_state:
        st.session_state.clean_df = df.copy() 
    df = st.session_state.clean_df  
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    st.subheader("Outlier Detection Method")
    method = st.selectbox("Choose a method", ["Z-Score", "Isolation Forest", "Local Outlier Factor (LOF)"])
    contamination = st.slider("Contamination (expected % of outliers)", 0.01, 0.3, 0.1, 0.01)
    z_thresh = st.slider("Z-Score Threshold", 2.0, 5.0, 3.0, 0.1)
    outlier_column = "Outlier"
    X = df[numeric_cols]
    y = df[target_column]
    if st.button("Detect Outliers"):
        if method == "Z-Score":
            z_scores = np.abs(zscore(X))
            outliers = (z_scores > z_thresh).any(axis=1)
            df[outlier_column] = np.where(outliers, "Yes", "No")
        elif method == "Isolation Forest":
            iso = IsolationForest(contamination=contamination, random_state=42)
            preds = iso.fit_predict(X)
            df[outlier_column] = np.where(preds == -1, "Yes", "No")
        elif method == "Local Outlier Factor (LOF)":
            lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
            preds = lof.fit_predict(X)
            df[outlier_column] = np.where(preds == -1, "Yes", "No")
        st.success(f"Outlier detection using {method} completed!")
        st.write("### Outlier Summary")
        st.dataframe(df[outlier_column].value_counts().rename_axis("Outlier").reset_index(name='Counts'))
        if df.isnull().values.any():
            st.warning("Dataset contains missing (NaN) values. Please handle them before proceeding.")
        else:
            st.subheader("PCA Visualization of Outliers")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            colors = df[outlier_column].map({"Yes": "red", "No": "green"})
            fig, ax = plt.subplots()
            ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.7)
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            ax.set_title("2D PCA Projection (Outliers Highlighted)")
            st.pyplot(fig)
    if st.button("Drop Outliers"):
        if outlier_column not in df.columns:
            st.warning(" Please detect outliers first.")
        else:
            st.session_state.clean_df = df[df[outlier_column] == "No"].drop(columns=[outlier_column])
            st.success("Outliers removed. Dataset updated for further operations.")
            st.dataframe(st.session_state.clean_df.head())
            csv = st.session_state.clean_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Cleaned CSV", csv, "cleaned_dataset.csv", "text/csv")
    target_col = st.selectbox("Select Target Column (y)", df.columns, key="target_select")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=['int64', 'float64']) 
    combined = pd.concat([X, y], axis=1)
    combined = combined.dropna()  
    X = combined.drop(columns=[target_col])
    y = combined[target_col]
    if X.shape[0] > 0 and y.shape[0] > 0 and not X.isnull().values.any() and not y.isnull().values.any():
        st.write("Shape of X:", X.shape)
        st.write("Shape of y:", y.shape)
        problem_type = type_of_target(y)
        st.subheader(f"Problem Type Detected: **{problem_type}**")
        if problem_type in ['binary', 'multiclass']:
            score_func = mutual_info_classif
            model = RandomForestClassifier()
        else:
            score_func = mutual_info_regression
            model = RandomForestRegressor()
        k = st.slider("Select number of top features (K)", min_value=1, max_value=len(X.columns), value=5, key="k1")
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        scores = selector.scores_
        feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': scores})
        feature_scores = feature_scores.sort_values(by='Score', ascending=False).head(k)
        st.subheader("Feature Importance via SelectKBest")
        st.dataframe(feature_scores)
        plt.figure(figsize=(10, 5))
        plt.barh(feature_scores['Feature'], feature_scores['Score'], color='skyblue')
        plt.xlabel("Importance Score")
        plt.title("Top K Feature Scores (SelectKBest)")
        plt.gca().invert_yaxis()
        st.pyplot(plt)
        model.fit(X, y)
        rf_scores = model.feature_importances_
        rf_feature_scores = pd.DataFrame({'Feature': X.columns, 'Importance': rf_scores})
        rf_feature_scores = rf_feature_scores.sort_values(by='Importance', ascending=False).head(k)
        st.subheader("Feature Importance via RandomForest")
        st.dataframe(rf_feature_scores)
        plt.figure(figsize=(10, 5))
        plt.barh(rf_feature_scores['Feature'], rf_feature_scores['Importance'], color='lightgreen')
        plt.xlabel("Feature Importance")
        plt.title("Top K Feature Importances (RandomForest)")
        plt.gca().invert_yaxis()
        st.pyplot(plt)
        if problem_type in ['binary', 'multiclass']:
            xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            xgb_model = XGBRegressor(verbosity=0)
            y_encoded = y 
        xgb_model.fit(X, y_encoded)
        xgb_scores = xgb_model.feature_importances_
        xgb_feature_scores = pd.DataFrame({'Feature': X.columns, 'Importance': xgb_scores})
        xgb_feature_scores = xgb_feature_scores.sort_values(by='Importance', ascending=False).head(k)
        st.subheader("Feature Importance via XGBoost")
        st.dataframe(xgb_feature_scores)
        plt.figure(figsize=(10, 5))
        plt.barh(xgb_feature_scores['Feature'], xgb_feature_scores['Importance'], color='orange')
        plt.xlabel("Feature Importance")
        plt.title("Top K Feature Importances (XGBoost)")
        plt.gca().invert_yaxis()
        st.pyplot(plt)
        merged = feature_scores.merge(rf_feature_scores, on='Feature', suffixes=('_SelectKBest', '_RF'))
        merged = merged.merge(xgb_feature_scores, on='Feature')
        merged.rename(columns={'Importance': 'Importance_XGB'}, inplace=True)
    else:
        st.warning("X or y is empty or contains only NaNs. Please check your data.")
    st.subheader("Auto ML Model Selector")
    st.markdown("Comparing baseline models using cross-validation")
    le = LabelEncoder()
    y = le.fit_transform(y)
    if problem_type in ['binary', 'multiclass']:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    }
        scoring = 'accuracy'
    else:
     models = {
         "Linear Regression": LinearRegression(),
         "Random Forest": RandomForestRegressor(),
         "XGBoost": XGBRegressor()
    }
     scoring = 'r2'
    scores_dict = {}
    for name, m in models.items():
        try:
            score = cross_val_score(m, X, y, cv=5, scoring=scoring)
            scores_dict[name] = np.mean(score)
        except Exception as e:
            st.write(f"Model {name} failed. Reason: {e}")
            scores_dict[name] = None
    st.subheader("Model Comparison")
    score_df = pd.DataFrame(list(scores_dict.items()), columns=['Model', 'Score']).dropna()
    st.dataframe(score_df)
    plt.figure(figsize=(8, 4))
    plt.bar(score_df['Model'], score_df['Score'], color='orchid')
    plt.ylabel(scoring)
    plt.title(f"Baseline {scoring} Comparison")
    st.pyplot(plt)
    st.subheader("FLAML AutoML Results")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    task = "classification" if problem_type in ['binary', 'multiclass'] else "regression"
    automl = AutoML()
    try:
        automl.fit(X_train=X_train, y_train=y_train, task=task, time_budget=60)
        st.markdown("FLAML AutoML Results")
        st.success(f"**Best Model:**\n\n{automl.model.estimator}")
        st.info(f"**Best Loss (Lower is Better):** `{automl.best_loss:.6f}`")
        st.markdown("Best Model Configuration:")
        for key, val in automl.best_config.items():
            st.markdown(f"- **{key}**: `{val}`")
    except Exception as e:
        st.error(f"FLAML AutoML failed. Reason: {e}")
    st.subheader("🔍 Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.subheader("Visualizations")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df_clean = df.copy()
    target_col = st.selectbox("Select target column for PCA", df.columns)
    with st.expander("Smart Visualizer: All-in-One Dataset View"):
        st.subheader(" Smart Visualizer: All-in-One View (No Target Required)")
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        if len(numeric_cols) >= 2:
            st.markdown(" Correlation Heatmap")
            fig = plt.figure(figsize=(10, 6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Correlation Heatmap")
            st.pyplot(fig)
        col2, col3 = st.columns(2)
        with col2:
            st.markdown(" Histograms (Numeric Features)")
            for col in numeric_cols:
                fig = plt.figure()
                sns.histplot(df[col], kde=True)
                plt.title(f"Histogram - {col}")
                st.pyplot(fig)
        with col3:
            st.markdown(" Countplots (Categorical Features)")
            for col in cat_cols:
                fig = plt.figure()
                sns.countplot(data=df, x=col)
                plt.title(f"Countplot - {col}")
                plt.xticks(rotation=45)
                st.pyplot(fig)
        col4, col5 = st.columns(2)
        with col4:
            if numeric_cols and cat_cols:
                st.markdown(" Boxplots (Numeric vs Category)")
                for num_col in numeric_cols[:3]:  # limit to 3
                    for cat_col in cat_cols[:2]:  # limit to 2
                        fig = plt.figure(figsize=(8, 5))
                        sns.boxplot(data=df, x=cat_col, y=num_col)
                        plt.title(f"{num_col} by {cat_col}")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
        with col5:
            if len(numeric_cols) >= 2:
                st.markdown(" Scatterplots (Pairwise)")
                pairs = list(zip(numeric_cols, numeric_cols[1:]))[:2]
                for x, y in pairs:
                    fig = plt.figure(figsize=(8, 5))
                    sns.scatterplot(data=df, x=x, y=y)
                    plt.title(f"{x} vs {y}")
                    st.pyplot(fig)
        st.subheader(" Dataset Feature Relationship Explorer and Regression Line ")
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        x_feature = st.selectbox("Select X-axis Feature", numeric_cols, key='x_feat')
        y_feature = st.selectbox("Select Y-axis Feature", numeric_cols, key='y_feat')
        fig, ax = plt.subplots()
        ax.scatter(df[x_feature], df[y_feature], alpha=0.7, color='teal', edgecolors='k')
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(f"{x_feature} vs {y_feature}")
        st.pyplot(fig)
        fig = plt.figure(figsize=(8, 5))
        sns.regplot(data=df, x=x_feature, y=y_feature, scatter_kws={'alpha':0.6})
        plt.title(f"{x_feature} vs {y_feature} with Regression Line")
        st.pyplot(fig)
    st.header(" KMeans Clustering on Original Dataset")
    if len(numeric_cols) >= 2:
        X_cluster = df[numeric_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        st.subheader(" Elbow Method to Suggest Best Number of Clusters")
        distortions = []
        K_range = range(1, 11)
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(X_scaled)
            distortions.append(km.inertia_)
        fig_elbow, ax_elbow = plt.subplots()
        ax_elbow.plot(K_range, distortions, marker='o')
        ax_elbow.set_xlabel('Number of clusters (K)')
        ax_elbow.set_ylabel('Inertia')
        ax_elbow.set_title('Elbow Method For Optimal K')
        st.pyplot(fig_elbow)
        st.subheader(" Choose Number of Clusters")
        n_clusters = st.slider("Select K (clusters)", min_value=2, max_value=10, value=3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        df['Cluster'] = cluster_labels
        st.subheader(" Cluster Sizes")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        st.dataframe(cluster_counts.rename("Count").reset_index().rename(columns={"index": "Cluster"}))
        st.subheader(" Cluster Centers (scaled values)")
        centers = pd.DataFrame(kmeans.cluster_centers_, columns=numeric_cols)
        centers['Cluster'] = centers.index
        st.dataframe(centers)
        st.subheader(" Cluster Scatter Plot (First Two Numeric Features)")
        fig_scatter, ax_scatter = plt.subplots()
        sns.scatterplot(x=X_cluster[numeric_cols[0]], y=X_cluster[numeric_cols[1]],
                        hue=cluster_labels, palette='tab10', alpha=0.7, ax=ax_scatter)
        ax_scatter.set_xlabel(numeric_cols[0])
        ax_scatter.set_ylabel(numeric_cols[1])
        ax_scatter.set_title("KMeans Clusters")
        st.pyplot(fig_scatter)
    else:
        st.warning(" Need at least two numeric columns for KMeans clustering.")
    st.header(" PCA Projection (2D)")
    if len(numeric_cols) >= 2:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numeric_cols])
        pca = PCA(n_components=2)
        components = pca.fit_transform(scaled_data)
        # Prepare DataFrame
        df_pca = pd.DataFrame(components, columns=['PCA1', 'PCA2'])
        df_pca['Target'] = df_clean[target_col].values
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(df_pca['PCA1'], df_pca['PCA2'],
                            c=pd.factorize(df_pca['Target'])[0],
                            cmap='viridis', alpha=0.7, edgecolors='w')
        labels = list(pd.Series(df_pca['Target']).unique())
        legend = ax.legend(*scatter.legend_elements(), title="Target", loc="best", labels=labels)
        ax.add_artist(legend)
        ax.set_title(" PCA Projection (2D)", fontsize=14)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        st.pyplot(fig)
        st.markdown(f"**Explained Variance:**")
        st.write(f"PCA1: {pca.explained_variance_ratio_[0]*100:.2f}%")
        st.write(f"PCA2: {pca.explained_variance_ratio_[1]*100:.2f}%")
    else:
        st.warning("PCA requires at least two numeric columns.")
    st.subheader("English Insight Generator")
    st.markdown("Auto-generated plain English insights")
    st.subheader("Insights from SelectKBest")
    for i, row in feature_scores.iterrows():
        st.write(f"Feature **{row['Feature']}** is highly predictive with a score of **{row['Score']:.2f}**.")
    st.subheader("Insights from RandomForest")
    for i, row in rf_feature_scores.iterrows():
        st.write(f"Feature **{row['Feature']}** is important in the Random Forest model with an importance of **{row['Importance']:.2f}**.")
    st.subheader("Best Performing Model")
    best_model = score_df.loc[score_df['Score'].idxmax()]
    st.write(f"The best model based on cross-validation is **{best_model['Model']}** with a score of **{best_model['Score']:.2f}**.")
    st.subheader("Redundant Feature Detection")
    st.markdown("""
    Detecting highly correlated features to avoid redundancy.  
    Columns with correlation > **0.9** are considered redundant and one of them will be suggested for removal.
    """)
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        st.warning("Not enough numeric features to compute correlations.")
    else:
        corr_matrix = numeric_df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
        if to_drop:
            st.success(f"{len(to_drop)} redundant features detected based on correlation > 0.9")
            st.write("Suggested Features to Drop:")
            st.dataframe(pd.DataFrame({"Feature": to_drop}))
            if st.button("Drop Redundant Features"):
                df = df.drop(columns=to_drop)
                st.success("Redundant features dropped successfully!")
            st.subheader("Correlation Heatmap:")
            fig, ax = plt.subplots(figsize=(10,8))
            cax = ax.matshow(corr_matrix, cmap='coolwarm')
            plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
            plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
            fig.colorbar(cax)
            st.pyplot(fig)
            st.subheader("Highly Correlated Pairs (>0.9):")
            high_corr_pairs = []
            for i in range(len(upper_tri.columns)):
                for j in range(i):
                    if upper_tri.iloc[j, i] > 0.9:
                        feature_1 = upper_tri.index[j]
                        feature_2 = upper_tri.columns[i]
                        corr_value = upper_tri.iloc[j, i]
                        high_corr_pairs.append((feature_1, feature_2, round(corr_value, 3)))
            high_corr_df = pd.DataFrame(high_corr_pairs, columns=["Feature 1", "Feature 2", "Correlation"])
            st.dataframe(high_corr_df)
        else:
            st.info("No redundant features detected (all correlations are below 0.9).")


    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    import gym
    from gym import spaces
    from sklearn.metrics import accuracy_score

    class DatasetEnv(gym.Env):
        def __init__(self, df, X, y, problem_type='classification'):
            super(DatasetEnv, self).__init__()
            self.df = df.copy()
            self.X = X
            self.y = y
            self.problem_type = problem_type
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(low=0, high=1, shape=(X.shape[1],), dtype=np.float32)
            self.step_count = 0

        def reset(self):
            self.step_count = 0
            obs = self._get_obs()
            if obs is None or np.any(np.isnan(obs)):
                obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs

        def _get_obs(self):
            obs = self.X.mean(axis=0).astype(np.float32)
            if np.any(np.isnan(obs)):
                print("[DEBUG] NaNs found in observation")
            return obs

        def step(self, action):
            reward = 0
            done = False
            if action == 1:
                model = RandomForestClassifier() if self.problem_type == 'classification' else RandomForestRegressor()
                try:
                    X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    score = accuracy_score(y_test, preds) if self.problem_type == 'classification' else model.score(X_test, y_test)
                    reward = score
                except Exception as e:
                    print(f"[ERROR] PPO model evaluation failed: {e}")
                    reward = -1
            self.step_count += 1
            done = self.step_count >= 1
            return self._get_obs(), reward, done, {}
        
        def seed(self, seed=None):
            np.random.seed(seed)
            return [seed]

    if st.button("Run Reinforcement Learning Optimization"):
        with st.spinner("Training PPO agent on your dataset..."):
            env = DatasetEnv(df, X.values, y, problem_type)
            vec_env = make_vec_env(lambda: env, n_envs=1)
            model = PPO("MlpPolicy", vec_env, verbose=0)
            model.learn(total_timesteps=1000)
            st.success("RL Agent trained successfully!")

            obs = env.reset()
            if obs is None or np.any(np.isnan(obs)):
                obs = np.zeros(env.observation_space.shape, dtype=np.float32)

            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)

            st.markdown("### PPO Agent Recommendation")
            st.write(f"Recommended Action: {'Apply ML Pipeline' if action == 1 else 'Skip Step'}")
            st.write(f"Estimated Reward: {reward:.4f}")

    # from train_agent import train_rl_agent
    # from rl_env import DatasetEnv

    # # Button to activate RL Optimization
    # if st.button("Run Reinforcement Learning Optimization"):
    #     with st.spinner("Training PPO agent on your dataset..."):
    #         model = train_rl_agent(df, problem_type)
    #         st.success("RL Agent trained successfully!")

    #         # Evaluate the best action from the agent
    #         env = DatasetEnv(df, problem_type)
    #         obs = env.reset()
    #         action, _ = model.predict(obs)
    #         def reset(self):
    #             self.step_count = 0
    #             obs = self._get_obs()
    #             if obs is None or np.any(np.isnan(obs)):
    #                 obs = np.zeros(self.observation_space.shape, dtype=np.float32)
    #             return obs
    #         def _get_obs(self):
    #             obs = self.X.mean(axis=0).astype(np.float32)
    #             if np.any(np.isnan(obs)):
    #                 print("[DEBUG] NaNs found in observation")  # Optional: log for debugging
    #             return obs
    #         obs = env.reset()
    #         if obs is None or np.any(np.isnan(obs)):
    #             obs = np.zeros(env.observation_space.shape, dtype=np.float32)

    #         action, _ = model.predict(obs)
    #         obs, reward, done, _ = env.step(action)
    #         st.markdown("### PPO Agent Recommendation")
    #         st.write(f"Recommended Action: {action}")
    #         st.write(f"Estimated Reward (Score+Efficiency): {reward:.4f}")

#     st.subheader("Download Insights as PDF")

#     # Function to generate PDF
#     if uploaded_file is not None and feature_scores is not None and rf_feature_scores is not None and xgb_feature_scores is not None and best_model is not None:
#     # Your PDF generation code here
#         class PDF(FPDF):
#             def header(self):
#                 self.set_font('Arial', 'B', 14)
#                 self.cell(0, 10, 'Dataset Insights Report', ln=True, align='C')
#                 self.ln(10)

#             def chapter_title(self, title):
#                 self.set_font('Arial', 'B', 12)
#                 self.cell(0, 10, title, ln=True)
#                 self.ln(5)

#             def chapter_body(self, body):
#                 self.set_font('Arial', '', 11)
#                 self.multi_cell(0, 10, body)
#                 self.ln()

#         # Create PDF
#         pdf = PDF()
#         pdf.add_page()

#         # Add content
#         pdf.chapter_title("General Dataset Info")
#         pdf.chapter_body(f"Number of Rows: {df.shape[0]}\nNumber of Columns: {df.shape[1]}")

#         pdf.chapter_title("Top Features (SelectKBest)")
#         for i, row in feature_scores.iterrows():
#             pdf.chapter_body(f"{row['Feature']}: Score {row['Score']:.2f}")

#         pdf.chapter_title("Top Features (Random Forest)")
#         for i, row in rf_feature_scores.iterrows():
#             pdf.chapter_body(f"{row['Feature']}: Importance {row['Importance']:.2f}")

#         pdf.chapter_title("Top Features (XGBoost)")
#         for i, row in xgb_feature_scores.iterrows():
#             pdf.chapter_body(f"{row['Feature']}: Importance {row['Importance']:.2f}")

#         pdf.chapter_title("Best Performing Model")
#         pdf.chapter_body(f"{best_model['Model']} with score {best_model['Score']:.2f}")

#         # Redundant features
#         if to_drop:
#             pdf.chapter_title("Redundant Features (Correlation > 0.9)")
#             pdf.chapter_body(", ".join(to_drop))
#         else:
#             pdf.chapter_title("Redundant Features (Correlation > 0.9)")
#             pdf.chapter_body("No highly redundant features detected.")

#         # Save PDF
#         pdf_output_path = "insights_report.pdf"
#         pdf.output(pdf_output_path)

#         # Make download button
#         with open(pdf_output_path, "rb") as f:
#             base64_pdf = base64.b64encode(f.read()).decode('utf-8')

#         href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="insights_report.pdf">📄 Download Insights Report</a>'
#         st.markdown(href, unsafe_allow_html=True)

#     # Example plot
#     plt.figure()
#     plt.plot([1, 2, 3, 4], [10, 20, 25, 30])
#     plt.title("Sample Plot")

#     # Save plot as an image
#     plt.savefig("plot_image.png")
#     plt.close()

#     class PDF(FPDF):
#         def chapter_title(self, title):
#             self.set_font('Arial', 'B', 14)
#             self.cell(0, 10, title, ln=1, align='L')
#             self.ln(5)

#         def chapter_body(self, body):
#             self.set_font('Arial', '', 12)
#             self.multi_cell(0, 10, body)
#             self.ln()

#     # Create PDF
#     pdf = PDF()
#     pdf.add_page()

#     # Add Insights
#     pdf.chapter_title("Dataset Insights")
#     pdf.chapter_body(f"Number of Rows: {df.shape[0]}\nNumber of Columns: {df.shape[1]}")

#     # Add the saved plot image
#     pdf.chapter_title("Graph Visualization")
#     pdf.image("plot_image.png", x=10, y=None, w=180)

#     # Save the PDF
#     pdf_output_path = "insights_report.pdf"
#     pdf.output(pdf_output_path)

#     # Read the PDF file
#     with open("insights_report.pdf", "rb") as f:
#         pdf_data = f.read()

#     # Streamlit download button
#     st.download_button(
#         label="Download Insights Report PDF",
#         data=pdf_data,
#         file_name="insights_report.pdf",
#         mime="application/pdf",
#     )
# # step 10
#     class MetaFeatureExtractor:
#         def __init__(self, df, target_column=None):
#             self.df = df
#             self.target_column = target_column
#         def extract_meta_features(self):
#             features = {}
#             # General Shape
#             features['num_rows'] = self.df.shape[0]
#             features['num_cols'] = self.df.shape[1]
#             # Missing Values
#             features['missing_values_total'] = self.df.isnull().sum().sum()
#             features['missing_values_percentage'] = features['missing_values_total'] / (features['num_rows'] * features['num_cols']) * 100
#             # Numeric Columns
#             numeric_df = self.df.select_dtypes(include=['int64', 'float64'])
#             features['num_numeric_cols'] = numeric_df.shape[1]
#             # Categorical Columns
#             categorical_df = self.df.select_dtypes(include=['object', 'category'])
#             features['num_categorical_cols'] = categorical_df.shape[1]
#             # Statistical Features (for numeric columns)
#             if not numeric_df.empty:
#                 features['mean_skewness'] = numeric_df.skew().mean()
#                 features['mean_kurtosis'] = numeric_df.kurtosis().mean()
#                 features['mean_variance'] = numeric_df.var().mean()
#                 features['mean_std'] = numeric_df.std().mean()
#                 features['mean_min'] = numeric_df.min().mean()
#                 features['mean_max'] = numeric_df.max().mean()
#                 features['mean_median'] = numeric_df.median().mean()
#             # Uniqueness
#             features['mean_unique_per_column'] = self.df.nunique().mean()
#             # Correlation
#             if numeric_df.shape[1] >= 2:
#                 corr_matrix = numeric_df.corr().abs()
#                 upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#                 features['average_correlation'] = upper_tri.stack().mean()
#             else:
#                 features['average_correlation'] = 0.0
#             # Target Related
#             if self.target_column and self.target_column in self.df.columns:
#                 features['num_unique_in_target'] = self.df[self.target_column].nunique()
#                 if is_numeric_dtype(self.df[self.target_column]):
#                     features['is_target_numeric'] = 1
#                 else:
#                     features['is_target_numeric'] = 0
#             else:
#                 features['num_unique_in_target'] = -1
#                 features['is_target_numeric'] = -1
#             # Column-wise Features
#             features['max_unique_in_columns'] = self.df.nunique().max()
#             features['min_unique_in_columns'] = self.df.nunique().min()
#             # Dataset Density (non-null / total
#             features['dataset_density'] = 1 - (self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1]))
#             # Number of Constant Columns (where all values are same)
#             features['constant_columns'] = sum(self.df.nunique() == 1)
#             return features
#     # Example usage
#     # Load any dataset
#     df = pd.read_csv("your_dataset.csv")
#     # Create the extractor object
#     extractor = MetaFeatureExtractor(df, target_column='target')  # if target exists
#     # Extract meta-features
#     meta_features = extractor.extract_meta_features()
#     # Convert to DataFrame
#     meta_df = pd.DataFrame([meta_features])
#     # Save to CSV (append if needed)
#     csv_path = "meta_features_collection.csv"
#     try:
#         old_df = pd.read_csv(csv_path)
#         final_df = pd.concat([old_df, meta_df], ignore_index=True)
#     except FileNotFoundError:
#         final_df = meta_df
#     final_df.to_csv(csv_path, index=False)
#     print("Meta-features saved successfully!")
#     # Load your dataset
#     df = pd.read_csv("your_dataset.csv")
#     # Initialize extractor
#     extractor = MetaFeatureExtractor(df, target_column='target')
#     # Extract features
#     features = extractor.extract_meta_features()
#     # Save / Print
#     print(features)
