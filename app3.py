import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2
from google import genai
from google.genai import types
from googletrans import Translator
import warnings
warnings.filterwarnings('ignore')

# =======================================================
# EARLY SESSION STATE INITIALIZATION
# =======================================================
if 'target_language_code' not in st.session_state:
    st.session_state.target_language_code = 'en'
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model_disease' not in st.session_state:
    st.session_state.model_disease = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'disease_features' not in st.session_state:
    st.session_state.disease_features = []
if 'uploaded_file_name_state' not in st.session_state:
    st.session_state.uploaded_file_name_state = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = "N/A"
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = None
if 'class_report' not in st.session_state:
    st.session_state.class_report = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if 'Date_Hierarchy_Created' not in st.session_state:
    st.session_state.Date_Hierarchy_Created = False
# --- EXPANDED PREDICTION INPUT FEATURES ---
PREDICTION_INPUT_FEATURES = [
    'region', 'crop_type', 'soil_moisture', 'soil_ph',
    'temperature', 'rainfall', 'humidity',
    'sunlight_hours', 'irrigation_method',
    'yield_kg', 'ndvi', 'ndwi', 'fertilizer_type', 'pesticide_used'
]
# ===========================
# Language Configuration
# ===========================
LANGUAGE_MAP = {
    'English': 'en',
    'Hindi': 'hi',
    'Spanish': 'es',
    'French': 'fr',
    'Chinese (Simplified)': 'zh-cn',
    'Marathi': 'mr'  
}
TRANSLATION_DICT = {
    "app_title": "Farmer Decision Support System",
    "app_subtitle": "Empowering Farmers with Data-Driven Insights for Smart Agriculture",
    "upload_file_title": "Upload your dataset (CSV)",
    "upload_warning": "Please upload a Smart Farming dataset to continue",
    "home_welcome": "Welcome to Smart Farming DSS",
    "home_analyze": "Analyze and visualize your agricultural data.",
    "home_detect": "Detect or predict crop diseases.",
    "home_cluster": "Cluster regions or crops for yield insights.",
    "home_chat": "Get instant advice from the AI Chatbot.",
    "home_decisions": "Make smart decisions for sustainability.",
    "home_tip": "Tip: Navigate through the menu to explore data analysis features.",
    "data_overview_header": "Dataset Overview",
    "preprocess_header": "Data Cleaning & Preprocessing",
    "filters_header": "Filters & Aggregations (OLAP Analysis)",
    "cluster_header": "Crop/Region Clustering",
    "classification_header": "Crop Disease Status Prediction",
    "graphs_header": "Custom Graphs",
    "chatbot_header": "AI Farmer Chatbot Assistant",
    "sidebar_language": "Select Language",
    "Navigation Menu": "Navigation Menu",
    "All Regions": "All Regions",
    "Mean": "Mean", "Sum": "Sum", "Max": "Max", "Min": "Min", "Count": "Count",
    "Predicted": "Predicted", "True": "True", "Actual": "Actual", "Frequency": "Frequency",
    "Feature": "Feature", "Chi-Squared Score": "Chi-Squared Score",
    # New keys for Graphs
    "Select Graph Type": "Select Graph Type",
    "Line": "Line", "Bar": "Bar", "Scatter": "Scatter", "Histogram": "Histogram",
    "Select X-axis (Categorical)": "Select X-axis (Categorical)",
    "Select Y-axis (Numeric Measure)": "Select Y-axis (Numeric Measure)",
    "Select Aggregation": "Select Aggregation",
    "Generate Graph": "Generate Graph",
    "Select Column (Numeric)": "Select Column (Numeric)",
    "Number of Bins": "Number of Bins",
    "Select Columns (X, Y)": "Select Columns (X, Y)",
    "Please select both an X-axis and a Y-axis.": "Please select both an X-axis and a Y-axis.",
    "Please select a column for the Histogram.": "Please select a column for the Histogram.",
    "Select at least two columns for Line/Scatter plot.": "Select at least two columns for Line/Scatter plot.",
    "Choose a graph type and select the appropriate columns to visualize your data.": "Choose a graph type and select the appropriate columns to visualize your data.",
    "Remove Missing Values": "Remove Missing Values",
    "Remove Duplicates": "Remove Duplicates",
    "Encode Categorical Columns": "Encode Categorical Columns",
    "Standardize Numeric Features": "Standardize Numeric Features",
    "Preview Before Cleaning": "Preview Before Cleaning",
    "Cleaned Dataset": "Cleaned Dataset",
    "The AI Farmer Assistant is thinking...": "The AI Farmer Assistant is thinking...",
    "Ask about your crop, soil, or any farming question...": "Ask about your crop, soil, or any farming question...",
    # Marathi additions for commonly used labels (improves speed/accuracy)
    "Please upload a dataset first.": "कृपया आधी डेटासेट अपलोड करा.",
    "Please upload a dataset first in the 'Data Upload' section.": "कृपया आधी 'डेटा अपलोड' विभागात डेटासेट अपलोड करा.",
    "Ask about your crop, soil, or any farming question...": "तुमच्या पिकाबद्दल, मातीबद्दल किंवा शेतीबद्दल काहीही विचारा...",
}

def T(key):
    text = TRANSLATION_DICT.get(key, key)
    if st.session_state.target_language_code == 'en':
        return text
    try:
        # translator setup is required here for use across functions
        translator = Translator()
        return translator.translate(text, dest=st.session_state.target_language_code).text
    except Exception:
        # Fallback to English if translation fails (e.g., API issues)
        return text

# ===========================
# Gemini AI Imports and Setup
# ===========================
@st.cache_resource(show_spinner=False)
def get_gemini_client():
    if "GEMINI_API_KEY" not in st.secrets:
        # st.warning("GEMINI_API_KEY not found in secrets.toml. Chatbot will be disabled.")
        return None
    try:
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        return None
client = get_gemini_client()
GEMINI_MODEL = "gemini-2.5-flash"

# ===========================
# Streamlit Page Setup
# ===========================
st.set_page_config(
    page_title=T("app_title"),
    page_icon="leaf",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================
# Dark Theme Styling
# ===========================
st.markdown(
    """
    <style>
    .stApp { background-color: #000000; color: #FFFFFF; font-family: 'Poppins', sans-serif; }
    h1, h2, h3, h4 { color: #FFD700; font-weight: 700; }
    [data-testid="stSidebar"] { background-color: #111111; color: #00FF00; }
    [data-testid="stSidebar"] * { color: #00FF00 !important; }
    div.stButton > button {
        background-color: #FFD700; color: #000000;
        border-radius: 12px; padding: 0.7em 1.4em;
        font-size: 1em; font-weight: 600;
        transition: all 0.2s ease-in-out;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.5);
    }
    div.stButton > button:hover { background-color: #E5C100; transform: scale(1.03); }
    .stDataFrame { background-color: #1E1E1E; border-radius: 10px; box-shadow: 0 0 15px rgba(0,0,0,0.7); }
    div[data-testid="stMetricValue"] { color: #00FF00; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===========================
# Data preprocessing for encoding and imputation
# ===========================
def preprocess_data(df, for_training=False):
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        imputer_num = SimpleImputer(strategy='mean')
        df_processed[numeric_cols] = imputer_num.fit_transform(df_processed[numeric_cols])
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        df_processed[col] = df_processed[col].fillna('Unknown').astype(str)
        if for_training:
            le = LabelEncoder()
            le.fit(df_processed[col])
            df_processed[col] = le.transform(df_processed[col])
            st.session_state.label_encoders[col] = le
        elif col in st.session_state.label_encoders:
            le = st.session_state.label_encoders[col]
            def safe_transform(x):
                try:
                    return le.transform([x])[0]
                except ValueError:
                    if 'Unknown' in le.classes_:
                        return le.transform(['Unknown'])[0]
                    return -1
            df_processed[col] = df_processed[col].apply(safe_transform)
        else:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            st.session_state.label_encoders[col] = le
    return df_processed

# ===========================
# Handle missing values (user-controlled) - Not fully used but kept for context
# ===========================
def impute_missing_values(df, num_strategy='mean', fill_value_num=None,
                         cat_strategy='most_frequent', fill_value_cat='Unknown'):
    df_imputed = df.copy()
    numeric_cols = df_imputed.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        imputer_num = SimpleImputer(strategy=num_strategy, fill_value=fill_value_num)
        df_imputed[numeric_cols] = imputer_num.fit_transform(df_imputed[numeric_cols])
    cat_cols = df_imputed.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy=cat_strategy, fill_value=fill_value_cat)
        df_imputed[cat_cols] = imputer_cat.fit_transform(df_imputed[cat_cols])
    return df_imputed

# ===========================
# Simplified RF training (with metrics)
# ===========================
@st.cache_data(show_spinner='Training Random Forest Model...')
def train_simple_rf_model(df, features, target='crop_disease_status'):
    try:
        if target not in df.columns:
            raise ValueError(f"No '{target}' column found for classification.")
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            raise ValueError("No suitable features found for training the model.")
        
        temp_df = df.copy()
        target_le = LabelEncoder()
        temp_df['target_encoded'] = target_le.fit_transform(
            temp_df[target].astype(str).fillna('Unknown'))
        st.session_state.label_encoders[target] = target_le
        
        status_text = st.empty()
        status_text.text("Preprocessing and feature engineering...")
        
        df_features = temp_df[available_features].copy()
        df_processed = preprocess_data(df_features, for_training=True)
        
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(df_processed)
        poly_feature_names = poly.get_feature_names_out(df_processed.columns)
        X_poly = pd.DataFrame(X_poly, columns=poly_feature_names)
        
        y = temp_df['target_encoded']
        
        k_best = min(20, X_poly.shape[1])
        selector = SelectKBest(chi2, k=k_best)
        X_new = selector.fit_transform(X_poly, y)
        selected_features_mask = selector.get_support()
        X_selected_features = X_poly.columns[selected_features_mask].tolist()
        X = pd.DataFrame(X_new, columns=X_selected_features)
        
        st.session_state.poly = poly
        st.session_state.selector = selector
        st.session_state.disease_features = X_selected_features
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        st.session_state.scaler = scaler
        
        try:
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(
                X_train_scaled, y_train)
        except ValueError:
            X_train_resampled, y_train_resampled = X_train_scaled, y_train
            
        status_text.text("Training Random Forest Classifier...")
        
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42)
        rf_model.fit(X_train_resampled, y_train_resampled)
        
        st.session_state.model_disease = rf_model
        st.session_state.best_model_name = 'RandomForest'
        
        y_pred = rf_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        class_rep = classification_report(y_test, y_pred, output_dict=True)
        
        st.session_state.accuracy = accuracy
        st.session_state.class_report = class_rep
        
        status_text.text("Training complete. Model ready for prediction.")
        return y_pred
    except Exception as e:
        st.error(f"Error training Random Forest model: {str(e)}")
        st.session_state.model_disease = None
        st.session_state.disease_features = []
        st.session_state.best_model_name = "N/A"
        st.session_state.accuracy = None
        st.session_state.class_report = None
        return None
    finally:
        status_text.empty()

# ===========================
# Disease classification (with scaling & feature selection)
# ===========================
def classify_disease(input_data, return_probabilities=False):
    try:
        if st.session_state.model_disease is None:
            return None
        
        input_df = pd.DataFrame([input_data])
        required_features = st.session_state.disease_features
        base_features = st.session_state.poly.feature_names_in_
        
        input_for_prep = input_df.copy()
        
        # Safely fill missing base features with appropriate defaults/means from the training data
        for feat in base_features:
            if feat not in input_for_prep.columns:
                if st.session_state.df is not None and feat in st.session_state.df.columns:
                    dtype = st.session_state.df[feat].dtype
                    if dtype in ['int64', 'float64']:
                        input_for_prep[feat] = st.session_state.df[feat].mean()
                    else:
                        mode_val = st.session_state.df[feat].mode()
                        input_for_prep[feat] = mode_val[0] if not mode_val.empty else 'Unknown'
                else:
                    input_for_prep[feat] = 0.0 if st.session_state.df is None else 'Unknown'

        # Apply preprocessing (encoding)
        input_processed = preprocess_data(input_for_prep[base_features].copy())
        
        # Apply polynomial features
        input_poly = st.session_state.poly.transform(input_processed)
        input_poly = pd.DataFrame(
            input_poly,
            columns=st.session_state.poly.get_feature_names_out(base_features)
        )
        
        # Select KBest features
        input_features = input_poly[required_features]
        
        # Scale features
        input_features_scaled = st.session_state.scaler.transform(input_features)
        
        model = st.session_state.model_disease
        predicted_class_idx = model.predict(input_features_scaled)[0]
        
        le = st.session_state.label_encoders.get('crop_disease_status')
        if le:
            predicted_disease = le.inverse_transform([predicted_class_idx])[0]
            
            if return_probabilities and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_features_scaled)[0]
                disease_probabilities = {
                    le.inverse_transform([i])[0]: float(prob)
                    for i, prob in enumerate(probabilities)
                }
                return predicted_disease, disease_probabilities
            
            return predicted_disease
        
        return predicted_class_idx
    except Exception as e:
        st.error(f"Error classifying disease: {str(e)}")
        return None

# ===========================
# OLAP operations
# ===========================
def perform_olap_operation(df, operation, dimensions, measures):
    df_temp = df.copy()
    numeric_measures = [m for m in measures if m in df_temp.select_dtypes(include=np.number).columns]
    
    # Pre-filter for Slice/Dice
    if operation in ['slice', 'dice'] and 'slice_dimension' in st.session_state and 'slice_values' in st.session_state:
        slice_dim = st.session_state.slice_dimension
        slice_vals = st.session_state.slice_values
        if slice_dim in df_temp.columns and slice_vals:
            df_temp = df_temp[df_temp[slice_dim].isin(slice_vals)]
            # For slice/dice, we might return the filtered data directly or aggregate
            if operation == 'slice':
                    # For pure slice, return subsetted data and simple mean aggregation
                    return df_temp.groupby(dimensions)[numeric_measures].mean().reset_index(), df_temp 
            # For dice, proceed to aggregation after filtering
        elif operation in ['slice', 'dice']: # If dimension/values are empty, still proceed to aggregation on full data if not slice
            pass

    try:
        if operation in ['slice', 'dice']:
            # Drill-down / Dice (filtered aggregation) is essentially a standard group-by
            if dimensions and numeric_measures:
                # Use mean for aggregation as a general default for numeric measures
                return df_temp.groupby(dimensions)[numeric_measures].mean().reset_index(), df_temp 
            elif dimensions:
                return df_temp.groupby(dimensions).size().to_frame(name='Count').reset_index(), df_temp
            else: # If no dimensions for grouping
                return df_temp[numeric_measures].mean().to_frame().T, df_temp

        elif operation == 'pivot':
            if len(dimensions) >= 2 and numeric_measures:
                pivot_df = df_temp.pivot_table(
                    index=dimensions[0], columns=dimensions[1],
                    values=numeric_measures[0], aggfunc='mean', fill_value=0)
                return pivot_df, df_temp
            return None, df_temp
            
        else:
            return None, df_temp
            
    except Exception as e:
        st.error(f"Error performing OLAP operation: {str(e)}")
        return None, None

# ===========================
# Clustering
# ===========================
def perform_clustering(df, selected_features, n_clusters=3):
    try:
        X = df[selected_features].copy()
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X_scaled)
        
        df_clustered = df.copy()
        df_clustered['Cluster'] = clusters
        return df_clustered
    except Exception as e:
        st.error(f"Error performing clustering: {str(e)}")
        return None

# ===========================
# Date hierarchy (optional)
# ===========================
def create_date_hierarchy(df):
    if 'Date_Hierarchy_Created' not in st.session_state or not st.session_state.Date_Hierarchy_Created:
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'timestamp' in c.lower()]
        if date_cols:
            try:
                date_col = next((c for c in date_cols if 'timestamp' in c.lower()), date_cols[0])
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df.dropna(subset=[date_col], inplace=True)
                if not df.empty:
                    df['Year'] = df[date_col].dt.year.astype('object')
                    df['Quarter'] = df[date_col].dt.quarter.astype('object')
                    df['Month'] = df[date_col].dt.month_name().str.slice(stop=3)
                    st.session_state.df = df.copy()
                    st.session_state.Date_Hierarchy_Created = True
            except Exception:
                st.session_state.Date_Hierarchy_Created = True
        else:
            st.session_state.Date_Hierarchy_Created = True


# ===========================
# Sidebar Navigation
# ===========================
selected_lang_name = st.sidebar.selectbox(
    T("sidebar_language"),
    options=list(LANGUAGE_MAP.keys()),
    key='language_selector_widget'
)
st.session_state.target_language_code = LANGUAGE_MAP[selected_lang_name]

menu_options = [
    "Home",
    "Data Upload",
    "Preprocessing",
    "Fliters",
    "Clustering",
    "Prediction",
    "Graphs",
    "AI Chatbot"
]
selected = st.sidebar.radio(T("Navigation Menu"), menu_options)


# ===========================
# Home
# ===========================
if selected == "Home" or not st.session_state.data_loaded:
    st.markdown(
        f"""
        <div style="background-color:#222222;padding:1.5em;border-radius:15px;text-align:center;margin-bottom:1em;">
            <h1><span style='color:#00FF00;'>&#x1f343;</span> {T("app_title")}</h1>
            <p style="color:#AAAAAA;">{T("app_subtitle")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if selected == "Home":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"### {T('home_welcome')}")
        if st.session_state.df is not None:
            st.write(f"Dataset loaded: **{st.session_state.get('uploaded_file_name', 'N/A')}**")
        st.write(f"- {T('home_analyze')}")
        st.write(f"- {T('home_detect')}")
        st.write(f"- {T('home_cluster')}")
        st.write(f"- {T('home_chat')}")
        st.write(f"- {T('home_decisions')}")
        st.info(T('home_tip'))
    with col2:
        # --- Farmer Image added to the home page ---
        st.image("famer.jpg", use_container_width=True)
        # ------------------------------------------

# ===========================
# Data Upload
# ===========================
elif selected == "Data Upload":
    st.header("Upload Your Dataset")
    st.markdown("---")
    uploaded_file = st.file_uploader(T("upload_file_title"), type=["csv"])
    if uploaded_file is not None:
        try:
            current_file_name = uploaded_file.name
            if st.session_state.uploaded_file_name_state != current_file_name:
                df = pd.read_csv(uploaded_file)
                df.columns = df.columns.str.lower()
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.session_state.uploaded_file_name = current_file_name
                st.session_state.uploaded_file_name_state = current_file_name
                st.session_state.Date_Hierarchy_Created = False
                st.session_state.df_clean = df.copy() # Initialize df_clean on upload
                create_date_hierarchy(st.session_state.df)
                # Reset ML states
                st.session_state.model_disease = None
                st.session_state.disease_features = []
                st.session_state.label_encoders = {}
                st.session_state.best_model_name = "N/A"
                st.session_state.accuracy = None
                st.session_state.class_report = None
                st.success("Dataset loaded successfully! Model training will be triggered on the 'Prediction' page.")
                st.rerun()
                
            df = st.session_state.df
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Number of Rows", df.shape[0])
            with col2:
                st.metric("Number of Columns", df.shape[1])
            st.subheader("Data Preview")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.session_state.data_loaded = False
    else:
        st.warning(T("upload_warning"))

# ===========================
# Preprocessing
# ===========================
elif selected == "Preprocessing":
    st.header(T("preprocess_header"))
    if st.session_state.df is None:
        st.warning(T("Please upload a dataset first."))
    else:
        if st.session_state.df_clean is None:
            st.session_state.df_clean = st.session_state.df.copy()
            
        df_clean = st.session_state.df_clean
        st.markdown(f"### {T('Preview Before Cleaning')}")
        st.dataframe(df_clean.head())
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button(T("Remove Missing Values"), key="btn_drop_na"):
                rows_before = len(df_clean)
                df_clean.dropna(inplace=True)
                rows_after = len(df_clean)
                st.session_state.df_clean = df_clean.copy()
                st.success(T(f"Missing values removed. Dropped {rows_before - rows_after} rows."))
        
        with col2:
            if st.button(T("Remove Duplicates"), key="btn_drop_dup"):
                rows_before = len(df_clean)
                df_clean.drop_duplicates(inplace=True)
                rows_after = len(df_clean)
                st.session_state.df_clean = df_clean.copy()
                st.success(T(f"Duplicate rows removed. Dropped {rows_before - rows_after} rows."))
        
        with col3:
            if st.button(T("Encode Categorical Columns"), key="btn_encode"):
                cat_cols = df_clean.select_dtypes(include="object").columns
                encoded_cols = []
                if len(cat_cols):
                    for col in cat_cols:
                        if col not in ['Month', 'Quarter', 'Year']:
                            try:
                                le = LabelEncoder()
                                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                                encoded_cols.append(col)
                            except Exception as e:
                                st.warning(f"Could not encode column {col}: {e}")
                    st.session_state.df_clean = df_clean.copy()
                    st.success(T(f"Encoded columns: {encoded_cols}"))
                else:
                    st.info(T("No categorical columns to encode."))
        
        with col4:
            if st.button(T("Standardize Numeric Features"), key="btn_standardize"):
                num_cols = df_clean.select_dtypes(include=np.number).columns
                if len(num_cols):
                    scaler = StandardScaler()
                    df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])
                    st.session_state.df_clean = df_clean.copy()
                    st.success(T("Numeric features standardized!"))
                else:
                    st.info(T("No numeric columns to standardize."))
        
        st.markdown(f"### {T('Cleaned Dataset')}")
        st.dataframe(df_clean.head())

# ===========================
# OLAP Operations
# ===========================
elif selected == "Fliters":
    st.header(T("filters_header"))
    if st.session_state.df is None:
        st.warning(T("Please upload a dataset first."))
    else:
        df = st.session_state.df.copy()
        df.columns = df.columns.str.lower()
        st.subheader("Operations")
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [c for c in categorical_cols if c not in
                            ['farm_id', 'sensor_id', 'id', 'sowing_date', 'harvest_date', 'timestamp']]

        col1, col2 = st.columns(2)
        
        operation = col1.selectbox(
            "Select Fliters:",
            ["slice", "dice", "pivot"], key="olap_op")

        # Dynamic dimension/measure selection based on operation
        if operation == "pivot":
            dimensions = col1.multiselect(
                "Select Dimensions (2 required):",
                options=categorical_cols,
                default=categorical_cols[:2] if len(categorical_cols) >= 2 else categorical_cols)
            measures = col2.selectbox(
                "Select Measure (1 required for values):",
                options=numeric_cols, default=numeric_cols[:1])
            measures = [measures] if measures else [] # Ensure measures is a list
        else:
            dimensions = col1.multiselect(
                "Select Dimensions (for grouping):",
                options=categorical_cols,
                default=categorical_cols[:2] if len(categorical_cols) >= 2 else categorical_cols)
            measures = col2.multiselect(
                "Select Measures (for aggregation):",
                options=numeric_cols,
                default=numeric_cols[:2])

        # Slice/Dice specific controls
        if operation in ["slice", "dice"] and categorical_cols:
            slice_dimension = st.selectbox(
                "Select dimension to filter (Slice/Dice):",
                categorical_cols, key="slice_dim")
            # Handle empty unique values case
            options_for_slice = df[slice_dimension].unique().tolist() if slice_dimension in df.columns else []
            slice_values = st.multiselect(
                f"Select {slice_dimension} values to keep:",
                options=options_for_slice,
                key="slice_vals")
            st.session_state.slice_dimension = slice_dimension
            st.session_state.slice_values = slice_values
        elif operation in ["slice", "dice"]:
            st.info("Cannot perform Slice/Dice: No categorical columns found.")


        if st.button("Apply Filter"):
            if not dimensions or not measures:
                st.warning("Please select at least one dimension and one measure.")
            else:
                with st.spinner("Performing Fliter..."):
                    result, original_data_subset = perform_olap_operation(
                        df, operation, dimensions, measures)
                    
                    if result is not None:
                        st.subheader("Fliter Result")
                        st.dataframe(result)
                        
                        # --- Visualization of Aggregated Results ---
                        if operation in ["slice", "dice"]:
                            if len(result) > 0 and len(dimensions) == 1 and len(measures) == 1:
                                st.subheader("Bar Chart Visualization")
                                
                                # Use the actual aggregated column name 
                                y_plot_col = result.columns[-1] 
                                
                                fig, ax = plt.subplots(figsize=(5, 3)) 
                                sns.barplot(data=result, x=dimensions[0], y=y_plot_col,
                                            palette="viridis", ax=ax)
                                ax.set_title(f"{operation.title()} - {dimensions[0]} by {measures[0]}",
                                             color="#FFFFFF")
                                ax.set_facecolor("#000000")
                                fig.patch.set_facecolor("#000000")
                                ax.set_xlabel(dimensions[0], color="#FFFFFF")
                                ax.set_ylabel(measures[0], color="#FFFFFF")
                                ax.tick_params(colors="#FFFFFF", axis='x', labelrotation=45)
                                ax.tick_params(colors="#FFFFFF", axis='y')
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                            elif len(dimensions) >= 2 and len(measures) == 1 and operation != "pivot":
                                st.subheader("Heatmap Visualization")
                                try:
                                    pivot_df = result.pivot(
                                        index=dimensions[0], columns=dimensions[1],
                                        values=measures[0])
                                    fig, ax = plt.subplots(figsize=(5, 4)) 
                                    sns.heatmap(pivot_df, annot=True, fmt=".1f",
                                                cmap="YlGnBu", linewidths=.5,
                                                linecolor='black', ax=ax)
                                    ax.set_title(
                                        f"Heatmap: {dimensions[0]} vs {dimensions[1]} by Mean {measures[0]}",
                                        color="#FFFFFF")
                                    ax.set_facecolor("#000000")
                                    fig.patch.set_facecolor("#000000")
                                    ax.tick_params(colors="#FFFFFF")
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.warning(f"Cannot create heatmap: {e}")
                            else:
                                st.info("Visualization supports single group-by or two group-bys with one measure.")
                                
                        elif operation == "pivot":
                            st.subheader("Pivot Table Heatmap")
                            if isinstance(result, pd.DataFrame):
                                fig, ax = plt.subplots(figsize=(5, 4)) 
                                sns.heatmap(result, annot=True, fmt=".1f",
                                            cmap="YlGnBu", linewidths=.5,
                                            linecolor='black', ax=ax)
                                ax.set_title(f"Pivot Heatmap - {dimensions[0]} vs {dimensions[1]}",
                                             color="#FFFFFF")
                                ax.set_facecolor("#000000")
                                fig.patch.set_facecolor("#000000")
                                ax.tick_params(colors="#FFFFFF")
                                st.pyplot(fig)
                                
                        elif operation == "slice":
                            st.subheader("Sliced Data Preview")
                            st.dataframe(original_data_subset.head(10))
                            if not original_data_subset.empty and measures:
                                st.subheader("Summary Statistics of Sliced Data")
                                st.dataframe(original_data_subset[measures].describe())
                    else:
                        st.error("Operation returned no result.")


# ===========================
# Clustering (MODIFIED TO MAX 2 FEATURES)
# ===========================
elif selected == "Clustering":
    st.header(T("cluster_header"))
    if st.session_state.df is None:
        st.warning(T("Please upload a dataset first."))
    else:
        df = st.session_state.df.copy()
        st.subheader("Cluster Crops/Regions")
        df.columns = df.columns.str.lower()
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        default_features = [f for f in ['rainfall', 'temperature', 'soil_moisture', 'yield_kg'] if f in numeric_cols]
        if len(default_features) < 2 and numeric_cols:
            default_features = numeric_cols[:2]
            
        # --- KEY MODIFICATION: max_selections=2 ---
        selected_features = st.multiselect(
            "Select features for clustering (max 2 for visualization):",
            options=numeric_cols, 
            default=default_features[:2],
            max_selections=2  
        )
        # ----------------------------------------
            
        n_clusters = st.slider("Number of clusters (K):", min_value=2, max_value=10, value=3)

        # Elbow Method
        if st.checkbox("Show Elbow Method for Optimal K"):
            if len(selected_features) >= 1:
                with st.spinner("Calculating inertia for Elbow Method..."):
                    X = df[selected_features]
                    imputer = SimpleImputer(strategy='mean')
                    X_imputed = imputer.fit_transform(X)
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_imputed)
                    inertia = []
                    k_range = range(1, 11)
                    
                    for k in k_range:
                        km = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_scaled)
                        inertia.append(km.inertia_)
                        
                    fig, ax = plt.subplots(figsize=(5, 3)) 
                    ax.plot(k_range, inertia, marker='o', color='#00FF00')
                    ax.set_title('Elbow Method', color="#FFFFFF")
                    ax.set_xlabel('Number of clusters (K)', color="#FFFFFF")
                    ax.set_ylabel('Inertia (Within-cluster sum of squares)', color="#FFFFFF")
                    ax.set_facecolor("#000000")
                    fig.patch.set_facecolor("#000000")
                    ax.tick_params(colors="#FFFFFF")
                    st.pyplot(fig)
            else:
                st.info("Select at least one feature to run the Elbow Method.")

        if st.button("Perform Clustering"):
            if len(selected_features) != 2: # Check for exactly 2 features (required for 2D plot)
                st.warning("Please select exactly 2 features for a clear cluster visualization.")
            else:
                with st.spinner("Performing clustering..."):
                    df_clustered = perform_clustering(df, selected_features, n_clusters)
                    
                    if df_clustered is not None:
                        cluster_sizes = df_clustered['Cluster'].value_counts().sort_index()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Cluster Distribution")
                            fig, ax = plt.subplots(figsize=(4, 4)) 
                            ax.pie(cluster_sizes.values,
                                   labels=[f"Cluster {i}" for i in cluster_sizes.index],
                                   autopct='%1.1f%%',
                                   colors=sns.color_palette("viridis", n_clusters))
                            ax.set_title("Cluster Distribution", color="#FFFFFF")
                            ax.set_facecolor("#000000")
                            fig.patch.set_facecolor("#000000")
                            st.pyplot(fig)
                            
                        with col2:
                            st.subheader("Cluster Means")
                            cluster_stats = df_clustered.groupby('Cluster')[selected_features] \
                                .mean().round(2).reset_index()
                            st.dataframe(cluster_stats)
                            
                        if len(selected_features) == 2:
                            st.subheader("2D Cluster Visualization")
                            fig, ax = plt.subplots(figsize=(5, 4)) 
                            sns.scatterplot(data=df_clustered,
                                            x=selected_features[0],
                                            y=selected_features[1],
                                            hue='Cluster',
                                            palette="viridis",
                                            ax=ax, s=100)
                            ax.set_title(f"{selected_features[0]} vs {selected_features[1]} - Clusters",
                                         color="#FFFFFF")
                            ax.set_facecolor("#000000")
                            fig.patch.set_facecolor("#000000")
                            ax.set_xlabel(selected_features[0], color="#FFFFFF")
                            ax.set_ylabel(selected_features[1], color="#FFFFFF")
                            ax.tick_params(colors="#FFFFFF")
                            ax.legend(facecolor="#111111", edgecolor="#444444", labelcolor="#FFFFFF")
                            st.pyplot(fig)
                            
                        st.subheader("Cluster Profiles")
                        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                        for cluster_id in range(n_clusters):
                            cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
                            with st.expander(f"Cluster {cluster_id} - {len(cluster_data)} records"):
                                col1_exp, col2_exp = st.columns(2)
                                with col1_exp:
                                    st.markdown("**Average Values:**")
                                    avg_vals = cluster_data[selected_features].mean().round(2)
                                    st.dataframe(avg_vals.rename("Value"))
                                with col2_exp:
                                    if len(categorical_cols) > 0:
                                        st.markdown("**Common Categories:**")
                                        for cat_col in [c for c in categorical_cols
                                                            if c not in ['farm_id', 'sensor_id', 'id']][:3]:
                                            if cat_col in df.columns and not cluster_data[cat_col].empty:
                                                top = cluster_data[cat_col].mode()
                                                st.text(f"{cat_col.title()}: {top.iloc[0] if not top.empty else 'N/A'}")
                                st.markdown("**Sample Records:**")
                                st.dataframe(cluster_data.head(3).drop('Cluster', axis=1))
                                
                        st.download_button(
                            label="Download Clustered Data",
                            data=df_clustered.to_csv(index=False).encode('utf-8'),
                            file_name=f"farming_clusters_k{n_clusters}.csv",
                            mime="text/csv"
                        )

# ===========================
# Prediction
# ===========================
elif selected == "Prediction":
    st.header(T("classification_header"))
    st.markdown("---")
    
    if st.session_state.df is None:
        st.warning(T("Please upload a dataset first in the 'Data Upload' section."))
    else:
        df = st.session_state.df.copy()
        df.columns = df.columns.str.lower()
        
        # --- Model Training/Checking ---
        if st.session_state.model_disease is None:
            st.info("Training Random Forest Classifier with advanced preprocessing...")
            train_simple_rf_model(df, PREDICTION_INPUT_FEATURES)
            
            if st.session_state.model_disease is None:
                st.error("Model training failed. Check data quality and ensure 'crop_disease_status' column exists.")
                st.stop()
            else:
                st.success("Random Forest Model Trained Successfully!")
                st.subheader("Model Performance")
                if st.session_state.accuracy is not None:
                    st.metric("Test Accuracy", f"{st.session_state.accuracy:.4f}")
                if st.session_state.class_report is not None:
                    class_rep_df = pd.DataFrame(st.session_state.class_report).transpose()
                    st.dataframe(class_rep_df)
        
        # --- Prediction Form ---
        required_features = st.session_state.disease_features
        base_features = [f for f in PREDICTION_INPUT_FEATURES if f in df.columns]
        
        feature_map = {
            'region': 'Region', 'crop_type': 'Crop Type',
            'soil_moisture': 'Soil Moisture (%)', 'soil_ph': 'Soil pH',
            'temperature': 'Temperature (°C)', 'rainfall': 'Rainfall (mm)',
            'humidity': 'Humidity (%)', 'sunlight_hours': 'Sunlight Hours',
            'irrigation_method': 'Irrigation Type', 'yield_kg': 'Yield (kg)',
            'ndvi': 'NDVI', 'ndwi': 'NDWI',
            'fertilizer_type': 'Fertilizer Type', 'pesticide_used': 'Pesticide Used'
        }
        
        with st.form("disease_prediction_form"):
            st.subheader("Input Farm Conditions")
            st.caption(f"Prediction uses the Random Forest model trained on {len(required_features)} optimized features.")
            
            cols = st.columns(3)
            input_data = {}
            
            for i, feat in enumerate(base_features):
                if feat not in df.columns:
                    continue
                
                with cols[i % 3]:
                    # Categorical input (Selectbox)
                    if df[feat].dtype == 'object':
                        options = sorted(df[feat].dropna().unique().tolist())
                        default_val = df[feat].mode()[0] if not df[feat].mode().empty else (options[0] if options else "Unknown")
                        input_data[feat] = st.selectbox(
                            feature_map.get(feat, feat).replace('_', ' ').title(),
                            options,
                            index=options.index(default_val) if default_val in options else 0,
                            key=f"pred{feat}")
                            
                    # Numeric input (Slider)
                    elif df[feat].dtype in ['int64', 'float64'] and not df[feat].dropna().empty:
                        min_val = float(df[feat].min())
                        max_val = float(df[feat].max())
                        default_val = float(df[feat].mean())
                        default_val = max(min_val, min(max_val, default_val)) # Clamp mean within min/max
                        
                        input_data[feat] = st.slider(
                            feature_map.get(feat, feat).replace('_', ' ').title(),
                            min_val, max_val, default_val,
                            key=f"pred{feat}")
                            
                    # Fallback for other or empty numerics
                    else:
                        input_data[feat] = st.number_input(
                            feature_map.get(feat, feat).replace('_', ' ').title(),
                            value=0.0, key=f"pred{feat}_num")

            st.markdown("---")
            submitted = st.form_submit_button("Predict Crop Disease Status")
            
            if submitted:
                with st.spinner("Predicting crop disease status using Random Forest..."):
                    # Prepare input data, ensuring all necessary features are present/imputed
                    actual_input = {}
                    for feat in base_features:
                         if feat in input_data:
                            actual_input[feat] = input_data[feat]
                         elif feat in df.columns:
                            mode_val = df[feat].mode()[0] if df[feat].dtype == 'object' and not df[feat].mode().empty else (
                                 df[feat].mean() if not df[feat].dropna().empty else 0)
                            actual_input[feat] = mode_val
                         else:
                            actual_input[feat] = 0.0 # Default if feature not in current form/data

                    prediction_result = classify_disease(actual_input, return_probabilities=True)
                    
                    if prediction_result is not None:
                        predicted_disease, probabilities = prediction_result
                        
                        st.success(f"Predicted Crop Disease Status: **{predicted_disease}**")
                        st.subheader("Prediction Probabilities")
                        
                        prob_df = pd.DataFrame.from_dict(
                            probabilities, orient='index', columns=['Probability']
                        ).sort_values('Probability', ascending=False)
                        st.dataframe(prob_df, use_container_width=True)
                    else:
                        st.error("Prediction failed. Please check inputs and model training.")
                        st.stop()

# ===========================
# Graphs (UPDATED FOR CATEGORICAL AXES & REDUCED SIZE)
# ===========================
elif selected == "Graphs":
    st.header(T("graphs_header"))
    if st.session_state.df is None:
        st.warning(T("Please upload a dataset first."))
    else:
        df = st.session_state.df.copy()
        df.columns = df.columns.str.lower()
        
        graph_map_reverse = {
            T("Line"): "Line", T("Bar"): "Bar",
            T("Scatter"): "Scatter", T("Histogram"): "Histogram"
        }
        
        graph_type = st.selectbox(
            T("Select Graph Type"),
            options=list(graph_map_reverse.keys()),
            key='graph_type_select'
        )
        graph_type_en = graph_map_reverse.get(graph_type, "Line")
        
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()

        # --- BAR PLOT: Categorical X-axis vs Numeric Y-axis ---
        if graph_type_en == "Bar":
            st.subheader(T("Bar") + " Chart Configuration")
            col1_g, col2_g = st.columns(2)
            
            x_col = col1_g.selectbox(T("Select X-axis (Categorical)"), 
                                     categorical_cols, key="bar_x")
            y_col = col2_g.selectbox(T("Select Y-axis (Numeric Measure)"), 
                                     numeric_cols, key="bar_y")
            agg_func_name = col2_g.selectbox(T("Select Aggregation"), 
                                             [T("Mean"), T("Sum"), T("Count")], key="bar_agg")
            
            if st.button(T("Generate Graph")):
                if x_col and y_col:
                    try:
                        # Map translated aggregation function back to English for pandas
                        agg_map = {T("Mean"): 'mean', T("Sum"): 'sum', T("Count"): 'count'}
                        pd_agg_func = agg_map.get(agg_func_name, 'mean')
                        
                        # Perform aggregation
                        if pd_agg_func == 'count':
                            plot_df = df.groupby(x_col)[y_col].count().reset_index(name=T("Count"))
                            plot_y_col = T("Count")
                        else:
                            plot_df = df.groupby(x_col)[y_col].agg(pd_agg_func).reset_index(name=f"{agg_func_name} of {y_col}")
                            plot_y_col = f"{agg_func_name} of {y_col}"

                        fig, ax = plt.subplots(figsize=(6, 4)) 
                        sns.barplot(data=plot_df, x=x_col, y=plot_y_col, palette="viridis", ax=ax)
                        
                        ax.set_title(f"{T('Bar')} Plot: {plot_y_col} by {x_col.title()}", color="#FFFFFF")
                        ax.set_facecolor("#000000")
                        fig.patch.set_facecolor("#000000")
                        ax.set_xlabel(x_col.title(), color="#FFFFFF")
                        ax.set_ylabel(plot_y_col, color="#FFFFFF")
                        ax.tick_params(colors="#FFFFFF", axis='x', labelrotation=45)
                        ax.tick_params(colors="#FFFFFF", axis='y')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Error generating Bar Plot: {e}")
                else:
                    st.warning(T("Please select both an X-axis and a Y-axis."))

        # --- HISTOGRAM: Single Numeric Column ---
        elif graph_type_en == "Histogram":
            st.subheader(T("Histogram") + " Configuration")
            col1_h, col2_h = st.columns(2)
            hist_col = col1_h.selectbox(T("Select Column (Numeric)"), 
                                         numeric_cols, key="hist_col")
            bins = col2_h.slider(T("Number of Bins"), 5, 100, 20)
            
            if st.button(T("Generate Graph")):
                if hist_col:
                    fig, ax = plt.subplots(figsize=(5, 3)) 
                    ax.hist(df[hist_col].dropna(), bins=bins, color="#FFD700")
                    ax.set_title(f"Histogram of {hist_col}", color="#FFFFFF")
                    ax.set_facecolor("#000000")
                    fig.patch.set_facecolor("#000000")
                    ax.set_xlabel(hist_col, color="#FFFFFF")
                    ax.set_ylabel(T("Frequency"), color="#FFFFFF")
                    ax.tick_params(colors="#FFFFFF")
                    st.pyplot(fig)
                else:
                    st.warning(T("Please select a column for the Histogram."))
        
        # --- LINE/SCATTER PLOT: Numeric Columns Only ---
        elif graph_type_en in ["Line", "Scatter"]:
            st.subheader(graph_type + " Plot Configuration")
            cols = st.multiselect(T("Select Columns (X, Y)"), 
                                     numeric_cols, 
                                     default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols[:1])
            
            if st.button(T("Generate Graph")):
                if len(cols) >= 2:
                    fig, ax = plt.subplots(figsize=(5, 3)) 
                    
                    if graph_type_en == "Line":
                        for c in cols[1:]:
                            ax.plot(df[cols[0]], df[c], label=c, linewidth=2)
                        
                    else: # Scatter
                        ax.scatter(df[cols[0]], df[cols[1]], color="#00FF00", s=50)
                        
                    ax.set_title(f"{graph_type_en} Plot: {cols[0]} vs {cols[1]}", color="#FFFFFF")
                    ax.legend(facecolor="#111111", edgecolor="#444444", labelcolor="#FFFFFF")
                    ax.set_facecolor("#000000")
                    fig.patch.set_facecolor("#000000")
                    ax.set_xlabel(cols[0], color="#FFFFFF")
                    ax.set_ylabel(cols[1], color="#FFFFFF")
                    ax.tick_params(colors="#FFFFFF")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning(T("Select at least two columns for Line/Scatter plot."))
        else:
            st.info(T("Choose a graph type and select the appropriate columns to visualize your data."))

# ===========================
# AI Chatbot
# ===========================
elif selected == "AI Chatbot":
    st.header(T("chatbot_header"))
    st.info(T("Ask me anything about farming, crop health, pest control, or general agriculture advice!"))
    
    if client is None:
        st.error("Gemini Client failed to initialize. Please check your GEMINI_API_KEY in secrets.toml.")
    else:
        SYSTEM_INSTRUCTION = (
            "You are an expert, helpful, and concise AI assistant for smart farming and agriculture. "
            "Your responses should be based on established agricultural best practices and data-driven insights. "
            "Keep your answers practical, actionable, and easy for a farmer to understand. "
            "Focus on giving advice on crop diseases, pest control, fertilization, irrigation scheduling, and soil health."
        )
        
        if "gemini_chat" not in st.session_state:
            config = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)
            st.session_state.gemini_chat = client.chats.create(
                model=GEMINI_MODEL, history=[], config=config)
            st.session_state.messages = []
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input(T("Ask about your crop, soil, or any farming question...")):
            current_lang = st.session_state.target_language_code
            
            # Translate user prompt to English for the model
            prompt_en = prompt
            if current_lang != 'en':
                try:
                    translator = Translator()
                    prompt_en = translator.translate(prompt, src=current_lang, dest='en').text
                except Exception:
                    prompt_en = prompt # Use original if translation fails
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            try:
                # Send English prompt to Gemini
                with st.spinner(T("The AI Farmer Assistant is thinking...")):
                    response = st.session_state.gemini_chat.send_message(prompt_en)
                    resp_text_en = response.text
                
                # Translate response back to the target language
                full_resp = resp_text_en
                if current_lang != 'en':
                    try:
                        translator = Translator()
                        full_resp = translator.translate(resp_text_en, dest=current_lang).text
                    except Exception:
                        full_resp = resp_text_en # Use English if translation fails
                
                # Display and save assistant response
                with st.chat_message("assistant"):
                    st.markdown(full_resp)
                st.session_state.messages.append({"role": "assistant", "content": full_resp})
                
            except Exception as e:
                st.error(T(f"An error occurred while connecting to the AI: {e}"))
