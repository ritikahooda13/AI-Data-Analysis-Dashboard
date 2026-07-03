import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Page Layout Configurations
st.set_page_config(page_title="AI Data Analysis Dashboard", layout="wide")

# UI Title Header
st.title("📊 AI Automated Data Analysis Dashboard")
st.write("Upload any structural dataset file to instantly execute advanced statistical profiling, filtering, and graphical exploration pipelines.")

# --- File Ingestion Engine ---
st.sidebar.header("Dataset Ingestion Panel")
uploaded_file = st.sidebar.file_uploader("Upload Target CSV File", type=["csv"])

# Target Mock Dataset Generator for immediate processing if no file is uploaded
@st.cache_data
def generate_fallback_data():
    np.random.seed(42)
    n_samples = 200
    mock_data = {
        'Product_ID': [f"PROD_{i:03d}" for i in range(1, n_samples + 1)],
        'Category': np.random.choice(['Electronics', 'Apparel', 'Home Decor', 'Fitness'], n_samples),
        'Sales_Revenue_USD': np.random.randint(500, 15000, n_samples),
        'Units_Sold': np.random.randint(10, 500, n_samples),
        'Customer_Satisfaction_Score': np.round(np.random.uniform(1.0, 5.0, n_samples), 2),
        'Region': np.random.choice(['North', 'East', 'West', 'South'], n_samples)
    }
    return pd.DataFrame(mock_data)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Dataset loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        df = generate_fallback_data()
else:
    st.sidebar.info("💡 Displaying baseline analytical matrix using automated analytical dataset.")
    df = generate_fallback_data()

# Identify Datatype Divisions
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=[object, 'category']).columns.tolist()

# --- Dynamic Navigation Panel ---
menu = st.sidebar.selectbox("Jump to Section", ["Structural Audit", "Dynamic Data Filters", "Statistical Profiling", "Advanced Visualizations"])

# --- SECTION 1: Structural Audit ---
if menu == "Structural Audit":
    st.header("📋 Baseline Data Structural Audit")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Extracted Rows", df.shape[0])
    c2.metric("Total Column Profiles", df.shape[1])
    c3.metric("Missing Matrix Values", df.isnull().sum().sum())
    
    st.write("### Data Table View (First 15 Records)")
    st.dataframe(df.head(15), use_container_width=True)
    
    st.write("### Internal Schema Datatypes & Completeness Profiles")
    schema_df = pd.DataFrame({
        'Datatype': df.dtypes.astype(str),
        'Non-Null Count': df.notnull().sum(),
        'Missing Fields': df.isnull().sum(),
        'Completeness Rate (%)': np.round((df.notnull().sum() / len(df)) * 100, 2)
    })
    st.dataframe(schema_df, use_container_width=True)

# --- SECTION 2: Dynamic Data Filters ---
elif menu == "Dynamic Data Filters":
    st.header("🔍 Multidimensional Dynamic Data Query Panel")
    st.write("Apply structural rules to slice the transactional dataset parameters:")
    
    filtered_df = df.copy()
    
    # Categorical Filters Setup
    if categorical_cols:
        st.write("#### Categorical Slicing Dimensions")
        cat_cols_to_filter = categorical_cols[:2] # Limit filters to top 2 for workspace clarity
        cols = st.columns(len(cat_cols_to_filter))
        for idx, col_name in enumerate(cat_cols_to_filter):
            with cols[idx]:
                unique_vals = df[col_name].unique().tolist()
                selected_vals = st.multiselect(f"Filter by {col_name}", unique_vals, default=unique_vals)
                filtered_df = filtered_df[filtered_df[col_name].isin(selected_vals)]
                
    # Numerical Ranges Setup
    if numerical_cols:
        st.write("#### Numerical Slicing Thresholds")
        num_cols_to_filter = numerical_cols[:2]
        cols = st.columns(len(num_cols_to_filter))
        for idx, col_name in enumerate(num_cols_to_filter):
            with cols[idx]:
                min_val = float(df[col_name].min())
                max_val = float(df[col_name].max())
                slider_range = st.slider(f"Range for {col_name}", min_val, max_val, (min_val, max_val))
                filtered_df = filtered_df[(filtered_df[col_name] >= slider_range[0]) & (filtered_df[col_name] <= slider_range[1])]

    st.write("### Filtered Output Results Matrix")
    st.metric("Records Matching Active Filtering Query", filtered_df.shape[0])
    st.dataframe(filtered_df, use_container_width=True)

# --- SECTION 3: Statistical Profiling ---
elif menu == "Statistical Profiling":
    st.header("🧮 Automated Statistical Summary Matrix")
    
    if numerical_cols:
        st.write("### Continuous Variable Distribution Summaries")
        st.dataframe(df.describe().fillna('-'), use_container_width=True)
        
        st.write("### Structural Feature Interaction Matrix (Correlation Values)")
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr()
            st.dataframe(corr_matrix.style.background_gradient(cmap='Blues').format(precision=3), use_container_width=True)
        else:
            st.info("Additional independent continuous variables are needed to render a structural covariance matrix.")
    else:
        st.warning("No continuous numeric values found within the current structural schema definitions.")

# --- SECTION 4: Advanced Visualizations ---
elif menu == "Advanced Visualizations":
    st.header("📊 Multi-Axis Graphic Chart Engine")
    
    if len(numerical_cols) >= 2:
        c1, c2 = st.columns(2)
        
        with c1:
            st.write("### Metric Interactions: Two-Variable Dispersion Spread")
            x_axis = st.selectbox("Select Independent Variable (X-Axis)", numerical_cols, index=0)
            y_axis = st.selectbox("Select Dependent Variable (Y-Axis)", numerical_cols, index=min(1, len(numerical_cols)-1))
            
            hue_axis = None
            if categorical_cols:
                hue_axis = st.selectbox("Select Categorical Grouping Label (Optional)", [None] + categorical_cols)
                
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=hue_axis, palette='coolwarm', ax=ax)
            plt.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
            
        with c2:
            st.write("### Distribution Profiling: Continuous Metric Densities")
            dist_col = st.selectbox("Select Profiling Target Continuous Column", numerical_cols, index=0)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df[dist_col], kde=True, color='#00B4D8', ax=ax)
            st.pyplot(fig)
            
        if len(numerical_cols) > 1:
            st.write("### System Correlation Matrix Graphic Representation")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='viridis', fmt=".2f", linewidths=0.5, ax=ax)
            st.pyplot(fig)
    else:
        st.warning("Insufficient multi-axis continuous parameters detected to map complex visual distributions.")
