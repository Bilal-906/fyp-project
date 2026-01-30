import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                               VotingClassifier, ExtraTreesClassifier, AdaBoostClassifier,
                               BaggingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Threat Detection System", layout="wide", initial_sidebar_state="expanded")

# Enhanced CSS styling
st.markdown("""
    <style>
    .main {background-color: rgba(255, 255, 255, 0.98); padding: 2rem; border-radius: 15px;}
    .title-box {text-align: center; background: linear-gradient(90deg, #11998e, #38ef7d); 
                  padding: 2rem 0; border-radius: 20px; color: white; margin-bottom: 2rem;
                  box-shadow: 0 10px 30px rgba(0,0,0,0.3);}
    .title-box h1 {font-size: 3rem; font-weight: 900; text-shadow: 3px 3px 6px rgba(0,0,0,0.4);}
    .dataset-box {background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                  color: white; padding: 2rem; border-radius: 15px; margin: 2rem 0;
                  box-shadow: 0 8px 25px rgba(0,0,0,0.3);}
    .stButton>button {background: linear-gradient(90deg, #11998e, #38ef7d); 
                        color: white; border-radius: 10px; padding: 1rem 2rem; 
                        font-size: 18px; font-weight: bold; width: 100%;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                        transition: all 0.3s;}
    .stButton>button:hover {transform: scale(1.05); box-shadow: 0 6px 20px rgba(0,0,0,0.3);}
    .metric-box {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  color: white; padding: 2rem; border-radius: 15px; text-align: center;
                  box-shadow: 0 8px 25px rgba(0,0,0,0.3); margin: 1rem 0;}
    .success-box {background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                   color: white; padding: 3rem; border-radius: 20px; text-align: center;
                   box-shadow: 0 10px 40px rgba(0,0,0,0.4); margin: 2rem 0;}
    .warning-box {background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                   color: white; padding: 2rem; border-radius: 15px;
                   box-shadow: 0 8px 25px rgba(0,0,0,0.3); margin: 1rem 0;}
    .info-box {background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white; padding: 1.5rem; border-radius: 15px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.3); margin: 1rem 0;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="title-box">
        <h1>🎯 CYBER THREAT DETECTION SYSTEM</h1>
        <p style="font-size: 1.5rem; font-weight: 700;">Advanced ML Training & Testing Platform</p>
    </div>
""", unsafe_allow_html=True)


def ultra_clean_data(df):
    """ULTRA-CLEAN data preprocessing for maximum accuracy"""
    try:
        st.write("### 🔥 ULTRA DATA CLEANING ENGINE")
        original_shape = df.shape
        cleaning_log = []
        
        # Step 1: Remove completely empty rows and columns
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        cleaning_log.append(f"✅ Removed empty rows/columns")
        
        # Step 2: Remove columns with excessive missing data (>70%)
        missing_threshold = 0.7
        missing_cols = df.columns[df.isnull().mean() > missing_threshold].tolist()
        if missing_cols:
            df = df.drop(columns=missing_cols)
            cleaning_log.append(f"✅ Dropped {len(missing_cols)} columns with >70% missing data")
        
        # Step 3: Handle infinite values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        cleaning_log.append(f"✅ Replaced infinite values with NaN")
        
        # Step 4: Smart imputation for missing values
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                mean_val = df[col].mean()
                std_val = df[col].std()
                
                # Use median for skewed distributions
                if std_val > 0 and abs(mean_val - median_val) > std_val * 0.5:
                    df[col].fillna(median_val, inplace=True)
                else:
                    df[col].fillna(mean_val, inplace=True)
        
        # Categorical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
        cleaning_log.append(f"✅ Imputed all missing values")
        
        # Step 5: Remove duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            cleaning_log.append(f"✅ Removed {duplicates} duplicate rows")
        
        # Step 6: Aggressive outlier handling with IQR method
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower = Q1 - 3.0 * IQR
                upper = Q3 + 3.0 * IQR
                df[col] = df[col].clip(lower, upper)
        cleaning_log.append(f"✅ Clipped outliers using 3.0×IQR method")
        
        # Step 7: Remove zero variance features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        zero_var = [col for col in numeric_cols if df[col].var() == 0]
        if zero_var:
            df = df.drop(columns=zero_var)
            cleaning_log.append(f"✅ Removed {len(zero_var)} zero-variance features")
        
        # Step 8: Remove low variance features
        low_var = [col for col in numeric_cols if col not in zero_var and 0 < df[col].var() < 0.001]
        if low_var:
            df = df.drop(columns=low_var)
            cleaning_log.append(f"✅ Removed {len(low_var)} low-variance features")
        
        # Step 9: Remove highly correlated features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr = [col for col in upper.columns if any(upper[col] > 0.98)]
            if high_corr:
                df = df.drop(columns=high_corr)
                cleaning_log.append(f"✅ Removed {len(high_corr)} highly correlated features (>0.98)")
        
        # Step 10: Handle skewed distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        skewed_count = 0
        for col in numeric_cols:
            skewness = df[col].skew()
            if abs(skewness) > 1.5:
                df[col] = np.log1p(df[col] - df[col].min() + 1)
                skewed_count += 1
        if skewed_count > 0:
            cleaning_log.append(f"✅ Log-transformed {skewed_count} highly skewed features")
        
        # Display cleaning log
        for log in cleaning_log:
            st.write(log)
        
        st.write(f"### ✅ CLEANING COMPLETE: {original_shape} → {df.shape}")
        return df
        
    except Exception as e:
        st.error(f"❌ Cleaning Error: {str(e)}")
        return df


def create_advanced_features(X, y):
    """Create advanced engineered features (optimized for speed)"""
    try:
        st.write("### ⚡ ADVANCED FEATURE ENGINEERING (FAST MODE)")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Select top features (reduced for speed)
            k_best = min(15, len(numeric_cols))
            selector = SelectKBest(mutual_info_classif, k=k_best)
            selector.fit(X[numeric_cols], y)
            top_features = [numeric_cols[i] for i in selector.get_support(indices=True)]
            
            st.write(f"🎯 Selected top {len(top_features)} features")
            
            feature_count = 0
            
            # Interaction features (reduced - only 3 combinations instead of 8)
            for i in range(min(5, len(top_features))):
                for j in range(i+1, min(i+3, len(top_features))):
                    col1, col2 = top_features[i], top_features[j]
                    
                    # Only multiplication (fastest)
                    X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                    feature_count += 1
            
            # Polynomial features (reduced - only square, no sqrt/log)
            for col in top_features[:5]:
                X[f'{col}_sq'] = X[col] ** 2
                feature_count += 1
            
            # Statistical aggregations (fast)
            X['row_mean'] = X[top_features].mean(axis=1)
            X['row_max'] = X[top_features].max(axis=1)
            feature_count += 2
            
            st.write(f"✅ Created {feature_count} engineered features (optimized)")
        
        return X
        
    except Exception as e:
        st.warning(f"⚠️ Feature engineering warning: {str(e)}")
        return X


# --- Helpers: safe, memory-friendly CSV reading ---------------------------------
@st.cache_data(show_spinner=False)
def read_csv_preview(path, nrows=10):
    """Read only the first nrows of a CSV using safe fallbacks to avoid huge memory use."""
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception:
        # fallback to a more permissive encoding/engine
        return pd.read_csv(path, nrows=nrows, encoding='cp1252', engine='python')


@st.cache_data(show_spinner=False)
def count_csv_lines(path):
    """Return number of records (excluding header) without loading full file into memory."""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            # subtract header
            count = sum(1 for _ in f) - 1
            return max(count, 0)
    except Exception:
        # fallback: try binary read (slightly faster sometimes)
        try:
            with open(path, 'rb') as f:
                count = sum(1 for _ in f) - 1
                return max(count, 0)
        except Exception:
            return None


@st.cache_data(show_spinner=False)
def compute_label_counts(path):
    """Compute value counts for 'label' column by reading CSV in chunks to save memory."""
    try:
        counts = {}
        for chunk in pd.read_csv(path, usecols=['label'], chunksize=200_000):
            vc = chunk['label'].value_counts()
            for k, v in vc.items():
                counts[k] = counts.get(k, 0) + int(v)
        return pd.Series(counts).sort_values(ascending=False)
    except Exception:
        return None


# Initialize session state
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'test_results' not in st.session_state:
    st.session_state.test_results = None
if 'model_artifacts' not in st.session_state:
    st.session_state.model_artifacts = None
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Authentication
if not st.session_state.authenticated:
    st.markdown("""
        <div class="info-box">
            <h2>🔒 Secure Login Required</h2>
            <p>Please login to access the Threat Detection System</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.form(key='login_form'):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<h3 style='text-align: center;'>Login</h3>", unsafe_allow_html=True)
            email = st.text_input("📧 Email", value="")
            password = st.text_input("🔑 Password", type="password", value="")
            # removed invalid width='stretch'
            login = st.form_submit_button("🚀 Login")
            
            if login:
                if email == "bilal123@gmail.com" and password == "bilal123":
                    st.session_state.authenticated = True
                    st.success("✅ Login Successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.")
else:
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    mode = st.sidebar.radio("Select Mode", ["📚 Train Model", "🧪 Test Model", "📊 View Results", "📥 Dataset"], index=0)
    
    if mode == "📚 Train Model":
        st.sidebar.markdown("---")
        st.sidebar.header("📁 Upload Dataset")
        
        # Dataset Selection Method
        dataset_method = st.sidebar.radio(
            "Choose Data Source",
            ["📤 Upload Your Data", "📱 Use Sample Data"],
            help="Select how you want to provide the training data"
        )
        
        if dataset_method == "📤 Upload Your Data":
            st.sidebar.markdown("""
                ### 📱 Mobile Upload Tips
                1. Tap the 'Browse files' button
                2. Select 'Files' or 'Documents'
                3. Navigate to your CSV file
                4. Select and confirm
                
                Supported format: CSV files
            """)
            uploaded_file = st.sidebar.file_uploader(
                "Upload Training CSV",
                type="csv",
                help="Upload your labeled training dataset. On mobile, make sure to allow file access.",
                accept_multiple_files=False
            )
        else:
            st.sidebar.markdown("### 📊 Choose Sample Dataset")
            sample_dataset = st.sidebar.radio(
                "Select a dataset to use:",
                ["Network Attack Dataset (Latest)", "Historical Network Data"],
                help="Choose which sample dataset to use for training"
            )
            
            if sample_dataset == "Network Attack Dataset (Latest)":
                st.sidebar.success("✅ Using latest network attack dataset")
                uploaded_file = "network_attack_dataset.csv"
            else:
                st.sidebar.success("✅ Using historical network data")
                uploaded_file = "old data.csv"
                
            st.sidebar.info("ℹ️ Sample datasets are pre-processed and ready for training")
        
        st.sidebar.markdown("---")
        st.sidebar.header("⚙️ Model Configuration")
        
        use_feature_eng = st.sidebar.checkbox(
            "🔥 Advanced Feature Engineering", 
            value=True,
            help="Creates interaction and polynomial features for higher accuracy"
        )
        
        balance_method = st.sidebar.selectbox(
            "⚖️ Class Balancing Method",
            ["SMOTETomek", "SMOTEENN", "SMOTE", "BorderlineSMOTE"],
            index=0,
            help="Technique to handle imbalanced classes"
        )
        
        test_size = st.sidebar.slider(
            "📊 Test Split Size",
            min_value=0.10,
            max_value=0.30,
            value=0.15,
            step=0.05,
            help="Percentage of data for testing (lower = more training data)"
        )
        
        n_estimators = st.sidebar.slider(
            "🌲 Number of Trees",
            min_value=50,
            max_value=300,
            value=100,
            step=50,
            help="Reduced range for faster training (50-300 trees)"
        )
        
        if uploaded_file:
            try:
                # Load data
                uploaded_file_path = None
                if isinstance(uploaded_file, str):  # Using sample data
                    df = pd.read_csv(uploaded_file)
                    # Remember which sample dataset was used so Test Model can reuse it
                    st.session_state['last_used_dataset'] = uploaded_file
                    uploaded_file_path = uploaded_file
                else:  # User uploaded file (Streamlit UploadedFile)
                    # Save uploaded file to disk so it can be reused later in the session
                    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploaded_datasets')
                    os.makedirs(uploads_dir, exist_ok=True)

                    # Create a safe filename and avoid collisions
                    orig_name = getattr(uploaded_file, 'name', f'uploaded_{int(time.time())}.csv')
                    safe_name = os.path.basename(orig_name)
                    dest_path = os.path.join(uploads_dir, safe_name)
                    if os.path.exists(dest_path):
                        base, ext = os.path.splitext(safe_name)
                        dest_path = os.path.join(uploads_dir, f"{base}_{int(time.time())}{ext}")

                    # Write uploaded bytes to disk
                    with open(dest_path, 'wb') as f:
                        try:
                            # UploadedFile supports getbuffer()
                            f.write(uploaded_file.getbuffer())
                        except Exception:
                            # Fallback to read()
                            f.write(uploaded_file.read())

                    # Load dataframe from saved file and remember path for reuse
                    df = pd.read_csv(dest_path)
                    st.session_state['last_used_dataset'] = dest_path
                    uploaded_file_path = dest_path
                
                st.markdown(f"""
                <div class="metric-box">
                    <h2>📊 Dataset Loaded Successfully</h2>
                    <p style="font-size: 1.8rem; margin: 0;">{df.shape[0]:,} rows × {df.shape[1]} columns</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📋 Total Rows", f"{df.shape[0]:,}")
                with col2:
                    st.metric("📊 Total Columns", df.shape[1])
                with col3:
                    st.metric("❌ Missing Values", f"{df.isnull().sum().sum():,}")
                with col4:
                    if 'label' in df.columns:
                        st.metric("🎯 Classes", df['label'].nunique())
                    else:
                        st.metric("🎯 Classes", "N/A")
                
                # Preview data
                with st.expander("🔍 Preview Dataset (First 30 Rows)"):
                    # Removed width='stretch' (Streamlit expects int or None)
                    st.dataframe(df.head(30))
                
                # Check for label column
                if 'label' not in df.columns:
                    st.markdown("""
                    <div class="warning-box">
                        <h3>❌ ERROR: Missing 'label' Column</h3>
                        <p>Your dataset must contain a column named 'label' with the target classes.</p>
                        <p><strong>Example:</strong> label column should have values like: benign, malware, attack, etc.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.stop()
                
                # Class distribution
                st.write("## 📊 Class Distribution Analysis")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    class_counts = df['label'].value_counts()
                    st.write("**Class Counts:**")
                    # Removed width='stretch'
                    st.dataframe(class_counts)
                    
                    # Check for class imbalance
                    imbalance_ratio = class_counts.max() / class_counts.min()
                    if imbalance_ratio > 10:
                        st.warning(f"⚠️ High class imbalance detected (ratio: {imbalance_ratio:.1f}:1). Balancing is recommended.")
                    else:
                        st.success(f"✅ Class balance is acceptable (ratio: {imbalance_ratio:.1f}:1)")
                
                with col2:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    class_counts.plot(kind='barh', ax=ax, color='#667eea', edgecolor='black')
                    ax.set_xlabel('Count', fontweight='bold', fontsize=12)
                    ax.set_ylabel('Class', fontweight='bold', fontsize=12)
                    ax.set_title('Class Distribution', fontweight='bold', fontsize=14)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Train button
                st.markdown("---")
                if st.button("🚀 START TRAINING"):
                    with st.spinner("🔄 Training in progress... Please wait..."):
                        try:
                            # Step 1: Clean data
                            df_clean = ultra_clean_data(df.copy())
                            
                            if df_clean.shape[0] == 0:
                                st.error("❌ No data remaining after cleaning!")
                                st.stop()
                            
                            # Step 2: Prepare features
                            drop_cols = [c for c in ['timestamp', 'flow_id', 'id', 'Unnamed: 0', 'index'] 
                                         if c in df_clean.columns]
                            
                            X = df_clean.drop(columns=drop_cols + ['label'], errors='ignore')
                            y = df_clean['label']
                            
                            st.write(f"**Features after cleaning:** {X.shape[1]}")
                            
                            # Step 3: Encode categorical features
                            cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
                            if cat_cols:
                                st.write(f"🔤 Encoding {len(cat_cols)} categorical columns...")
                                X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
                                st.write(f"**Features after encoding:** {X.shape[1]}")
                            
                            # Step 4: Feature engineering
                            if use_feature_eng:
                                X = create_advanced_features(X, y)
                                st.write(f"**Features after engineering:** {X.shape[1]}")
                            
                            # Step 5: Encode labels
                            le = None
                            if y.dtype == 'object':
                                le = LabelEncoder()
                                y_encoded = le.fit_transform(y)
                                st.write(f"✅ Encoded {len(le.classes_)} classes: {', '.join(le.classes_)}")
                            else:
                                y_encoded = y.values
                            
                            # Step 6: Split data
                            st.write(f"📊 Splitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y_encoded, 
                                test_size=test_size, 
                                random_state=42, 
                                stratify=y_encoded
                            )
                            
                            # Step 7: Scale features (single scaler for speed)
                            st.write("⚖️ Scaling features with StandardScaler...")
                            scaler = StandardScaler()
                            
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            
                            # Step 8: Balance classes
                            st.write(f"⚖️ Balancing classes using {balance_method}...")
                            
                            if balance_method == "SMOTETomek":
                                # Some samplers don't accept n_jobs in all versions; omit to be compatible
                                balancer = SMOTETomek(random_state=42)
                            elif balance_method == "SMOTEENN":
                                balancer = SMOTEENN(random_state=42)
                            elif balance_method == "BorderlineSMOTE":
                                balancer = BorderlineSMOTE(random_state=42)
                            else:
                                balancer = SMOTE(random_state=42)

                            # fit_resample may return 2- or 3-element tuples depending on version;
                            # index explicitly to avoid tuple-unpack type errors from static analysis.
                            res = balancer.fit_resample(X_train_scaled, y_train)
                            X_train_bal = res[0]
                            y_train_bal = res[1]
                            st.write(f"✅ Balanced: {len(y_train):,} → {len(y_train_bal):,} samples")
                            
                            # Step 9: Train models
                            st.write("## 🚀 TRAINING ENSEMBLE MODELS")
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            models = {
                                'Random Forest': RandomForestClassifier(
                                    n_estimators=n_estimators,
                                    max_depth=15,
                                    min_samples_split=5,
                                    min_samples_leaf=2,
                                    max_features='sqrt',
                                    bootstrap=True,
                                    class_weight='balanced_subsample',
                                    criterion='gini',
                                    random_state=42,
                                    n_jobs=-1
                                ),
                                'Extra Trees': ExtraTreesClassifier(
                                    n_estimators=n_estimators,
                                    max_depth=15,
                                    min_samples_split=5,
                                    min_samples_leaf=2,
                                    max_features='sqrt',
                                    bootstrap=True,
                                    class_weight='balanced_subsample',
                                    criterion='gini',
                                    random_state=42,
                                    n_jobs=-1
                                ),
                                'Gradient Boosting': GradientBoostingClassifier(
                                    n_estimators=min(150, n_estimators),
                                    learning_rate=0.1,
                                    max_depth=8,
                                    min_samples_split=5,
                                    subsample=0.9,
                                    max_features='sqrt',
                                    random_state=42,
                                    validation_fraction=0.1,
                                    n_iter_no_change=5
                                ),
                                'AdaBoost': AdaBoostClassifier(
                                    n_estimators=min(150, n_estimators),
                                    learning_rate=0.8,
                                    algorithm='SAMME',
                                    random_state=42
                                ),
                                'Bagging Ensemble': BaggingClassifier(
                                    estimator=RandomForestClassifier(
                                        n_estimators=50,
                                        max_depth=15,
                                        class_weight='balanced',
                                        random_state=42,
                                        n_jobs=-1
                                    ),
                                    n_estimators=30,
                                    max_samples=0.8,
                                    max_features=0.8,
                                    bootstrap=True,
                                    random_state=42,
                                    n_jobs=-1
                                )
                            }
                            
                            results = {}
                            trained_models = []
                            total_models = len(models)
                            
                            for idx, (name, model) in enumerate(models.items()):
                                status_text.text(f"⚙️ Training {name}... ({idx+1}/{total_models})")
                                
                                model.fit(X_train_bal, y_train_bal)
                                y_pred = model.predict(X_test_scaled)
                                
                                results[name] = {
                                    'accuracy': accuracy_score(y_test, y_pred),
                                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                                    'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                                }
                                
                                trained_models.append((name, model))
                                progress_bar.progress((idx + 1) / total_models)
                            
                            # Voting Ensemble
                            status_text.text(f"⚙️ Building Voting Ensemble... ({len(models)+1}/{total_models})")
                            voting_model = VotingClassifier(
                                estimators=trained_models,
                                voting='soft',
                                weights=[2, 2, 1, 1],
                                n_jobs=-1
                            )
                            voting_model.fit(X_train_bal, y_train_bal)
                            
                            y_pred_voting = voting_model.predict(X_test_scaled)
                            results['VOTING ENSEMBLE'] = {
                                'accuracy': accuracy_score(y_test, y_pred_voting),
                                'precision': precision_score(y_test, y_pred_voting, average='weighted', zero_division=0),
                                'recall': recall_score(y_test, y_pred_voting, average='weighted', zero_division=0),
                                'f1_score': f1_score(y_test, y_pred_voting, average='weighted', zero_division=0)
                            }
                            progress_bar.progress((len(models) + 1) / total_models)
                            
                            # Stacking Ensemble
                            status_text.text(f"⚙️ Building Stacking Ensemble... ({len(models)+2}/{total_models})")
                            stacking_model = StackingClassifier(
                                estimators=trained_models,
                                final_estimator=LogisticRegression(
                                    max_iter=2000,
                                    random_state=42,
                                    C=0.5,
                                    solver='lbfgs',
                                    class_weight='balanced'
                                ),
                                cv=5,
                                n_jobs=-1,
                                passthrough=False
                            )
                            stacking_model.fit(X_train_bal, y_train_bal)
                            
                            y_pred_stack = stacking_model.predict(X_test_scaled)
                            results['STACKING ENSEMBLE'] = {
                                'accuracy': accuracy_score(y_test, y_pred_stack),
                                'precision': precision_score(y_test, y_pred_stack, average='weighted', zero_division=0),
                                'recall': recall_score(y_test, y_pred_stack, average='weighted', zero_division=0),
                                'f1_score': f1_score(y_test, y_pred_stack, average='weighted', zero_division=0)
                            }
                            progress_bar.progress(1.0)
                            status_text.text("✅ Training Complete!")
                            
                            # Select best model
                            best_model_name = max(results, key=lambda k: results[k]['accuracy'])
                            
                            if best_model_name == 'VOTING ENSEMBLE':
                                best_model = voting_model
                                best_pred = y_pred_voting
                            elif best_model_name == 'STACKING ENSEMBLE':
                                best_model = stacking_model
                                best_pred = y_pred_stack
                            else:
                                best_model = dict(trained_models)[best_model_name]
                                best_pred = best_model.predict(X_test_scaled)
                            

                            # Store results
                            st.session_state.trained_model = best_model
                            st.session_state.test_results = {
                                'y_test': y_test,
                                'y_pred': best_pred,
                                'results': results,
                                'label_encoder': le,
                                'best_model_name': best_model_name
                            }
                            st.session_state.model_artifacts = {
                                'model': best_model,
                                'scaler': scaler,
                                'drop_cols': drop_cols,
                                'cat_cols': cat_cols,
                                'label_encoder': le,
                                'feature_names': X.columns.tolist()
                            }
                            # Persist which dataset was used for training (if any) so Test mode can reuse it
                            try:
                                # If we saved the uploaded file during load, use that path
                                training_dataset_path = uploaded_file_path  # may be None if not set
                            except NameError:
                                training_dataset_path = None

                            if training_dataset_path is None and isinstance(uploaded_file, str):
                                training_dataset_path = uploaded_file

                            st.session_state.model_artifacts['training_dataset_path'] = training_dataset_path
                            
                            # Display results
                            st.markdown("---")
                            st.write("## 🏆 TRAINING RESULTS")
                            
                            results_df = pd.DataFrame({
                                'Model': list(results.keys()),
                                'Accuracy (%)': [r['accuracy'] * 100 for r in results.values()],
                                'Precision (%)': [r['precision'] * 100 for r in results.values()],
                                'Recall (%)': [r['recall'] * 100 for r in results.values()],
                                'F1-Score (%)': [r['f1_score'] * 100 for r in results.values()]
                            }).round(2)
                            
                            results_df = results_df.sort_values('Accuracy (%)', ascending=False)
                            
                            # removed width='stretch' (must be int or omitted)
                            st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))
                            
                            best_acc = results[best_model_name]['accuracy'] * 100
                            best_prec = results[best_model_name]['precision'] * 100
                            best_rec = results[best_model_name]['recall'] * 100
                            best_f1 = results[best_model_name]['f1_score'] * 100
                            
                            # Show accuracy status
                            if best_acc >= 90:
                                st.markdown(f"""
                                <div class="success-box">
                                    <h1>🎉 EXCELLENT! TARGET ACHIEVED!</h1>
                                    <h2>Best Model: {best_model_name}</h2>
                                    <h1 style="font-size: 5rem; margin: 1rem 0;">{best_acc:.2f}%</h1>
                                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 2rem; margin-top: 2rem;">
                                        <div>
                                            <h3>Precision</h3>
                                            <h2 style="font-size: 2rem;">{best_prec:.2f}%</h2>
                                        </div>
                                        <div>
                                            <h3>Recall</h3>
                                            <h2 style="font-size: 2rem;">{best_rec:.2f}%</h2>
                                        </div>
                                        <div>
                                            <h3>F1-Score</h3>
                                            <h2 style="font-size: 2rem;">{best_f1:.2f}%</h2>
                                        </div>
                                    </div>
                                    <p style="font-size: 1.2rem; margin-top: 2rem;">✅ Model is ready for deployment!</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="warning-box">
                                    <h2>⚠️ Current Accuracy: {best_acc:.2f}%</h2>
                                    <h3>Best Model: {best_model_name}</h3>
                                    <p style="font-size: 1.1rem; margin-top: 1rem;">Below 80% target. Consider these improvements:</p>
                                    <ul style="text-align: left; display: inline-block;">
                                        <li>✅ Ensure dataset has clear, distinct patterns between classes</li>
                                        <li>✅ Check for data quality (consistent labels, no noise)</li>
                                        <li>✅ Collect more diverse training samples</li>
                                        <li>✅ Enable Advanced Feature Engineering</li>
                                        <li>✅ Increase number of trees to 1000-1500</li>
                                        <li>✅ Try different balancing methods</li>
                                        <li>✅ Reduce test size to 10% for more training data</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Save model button
                            st.markdown("---")
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                # removed width='stretch'
                                if st.button("💾 SAVE MODEL"):
                                    try:
                                        joblib.dump(st.session_state.model_artifacts, 'trained_threat_model.pkl')
                                        st.success("✅ Model saved successfully as 'trained_threat_model.pkl'")
                                    except Exception as e:
                                        st.error(f"❌ Error saving model: {str(e)}")
                            
                        except Exception as e:
                            st.error(f"❌ Training Error: {str(e)}")
                            import traceback
                            with st.expander("🔍 View Error Details"):
                                st.code(traceback.format_exc())
                
            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")
                import traceback
                with st.expander("🔍 View Error Details"):
                    st.code(traceback.format_exc())
    
    elif mode == "🧪 Test Model":
        st.write("## 🧪 TEST YOUR TRAINED MODEL")
        
        if st.session_state.trained_model is None:
            st.markdown("""
            <div class="info-box">
                <h3>⚠️ No Trained Model Available</h3>
                <p>Please train a model first or load a saved model below.</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_model = st.file_uploader("📂 Upload Saved Model (.pkl)", type=['pkl'])
            
            if uploaded_model:
                try:
                    st.session_state.model_artifacts = joblib.load(uploaded_model)
                    st.session_state.trained_model = st.session_state.model_artifacts['model']
                    st.success("✅ Model loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
        else:
            st.success("✅ Trained model is ready for testing!")
            
            # Allow re-using the dataset used during training (if available) or uploading a new test CSV
            # Safely obtain a string path to the training dataset if available
            training_path = None
            ma = st.session_state.get('model_artifacts')
            if ma and isinstance(ma.get('training_dataset_path'), str):
                training_path = ma.get('training_dataset_path')
            elif st.session_state.get('last_used_dataset') and isinstance(st.session_state.get('last_used_dataset'), str):
                training_path = st.session_state.get('last_used_dataset')

            if training_path:
                test_choice = st.radio("📁 Test data source", ["Use training dataset", "Upload test CSV"], index=0)
            else:
                test_choice = "Upload test CSV"

            test_df = None
            if test_choice == "Use training dataset":
                if training_path:
                    try:
                        # Check file size and avoid auto-loading very large files on constrained hosts
                        try:
                            tp_size = os.path.getsize(training_path) / (1024 * 1024)
                        except Exception:
                            tp_size = None

                        if tp_size and tp_size > 20:
                            st.warning(f"⚠️ Training dataset is large ({tp_size:.1f} MB). Loading it may exceed memory limits.")
                            load_large = st.checkbox("I understand and want to load the full training dataset for testing")
                            if not load_large:
                                st.info("Please upload a smaller test CSV or enable the checkbox to load the full dataset.")
                                test_choice = "Upload test CSV"
                                test_df = None
                            else:
                                test_df = pd.read_csv(training_path)
                                st.success(f"✅ Loaded training dataset for testing: {os.path.basename(training_path)}")
                        else:
                            test_df = pd.read_csv(training_path)
                            st.success(f"✅ Loaded training dataset for testing: {os.path.basename(training_path)}")
                    except Exception as e:
                        st.error(f"❌ Failed to load training dataset: {str(e)}")
                        st.info("Please upload a test CSV file instead.")
                        test_choice = "Upload test CSV"
                else:
                    st.error("❌ No valid training dataset path available. Please upload a test CSV file.")
                    test_choice = "Upload test CSV"

            if test_choice == "Upload test CSV":
                test_file = st.file_uploader("📁 Upload Test Dataset (CSV)", type="csv")
                if test_file:
                    try:
                        test_df = pd.read_csv(test_file)
                    except Exception as e:
                        st.error(f"❌ Error loading test file: {str(e)}")
                        test_df = None

            if test_df is not None:
                try:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>📊 Test Dataset Loaded</h3>
                        <p style="font-size: 1.5rem;">{test_df.shape[0]:,} rows × {test_df.shape[1]} columns</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("🔍 Preview Test Data (First 20 Rows)"):
                        # removed width='stretch'
                        st.dataframe(test_df.head(20))
                    
                    if st.button("🚀 RUN PREDICTION", width='stretch'):
                        with st.spinner("🔄 Making predictions..."):
                            try:
                                artifacts = st.session_state.model_artifacts

                                # Guard: ensure artifacts are present and valid
                                if artifacts is None:
                                    st.error("❌ No model artifacts available. Please train a model first or load a saved model.")
                                    st.stop()

                                # Check for labels
                                has_labels = 'label' in test_df.columns
                                
                                if has_labels:
                                    # Check for missing values in labels
                                    if test_df['label'].isnull().any():
                                        missing_count = test_df['label'].isnull().sum()
                                        st.warning(f"⚠️ Found {missing_count} missing values in 'label' column. Removing these rows for evaluation.")
                                        # Keep track of original indices
                                        valid_indices = test_df['label'].notna()
                                        test_df = test_df[valid_indices].reset_index(drop=True)
                                        
                                        if len(test_df) == 0:
                                            st.error("❌ All labels are missing! Cannot evaluate model performance.")
                                            has_labels = False
                                            y_true = None
                                            test_df_clean = test_df.copy()
                                        else:
                                            y_true = test_df['label'].copy()
                                            test_df_clean = test_df.drop(columns=['label'])
                                            st.info(f"✅ Using {len(test_df)} samples with valid labels for evaluation.")
                                    else:
                                        y_true = test_df['label'].copy()
                                        test_df_clean = test_df.drop(columns=['label'])
                                else:
                                    y_true = None
                                    test_df_clean = test_df.copy()
                                
                                # Drop unnecessary columns
                                test_df_clean = test_df_clean.drop(columns=artifacts['drop_cols'], errors='ignore')
                                
                                # Encode categorical
                                if artifacts['cat_cols']:
                                    for col in artifacts['cat_cols']:
                                        if col in test_df_clean.columns:
                                            test_df_clean = pd.get_dummies(test_df_clean, columns=[col], drop_first=True)
                                
                                # Align features
                                for col in artifacts['feature_names']:
                                    if col not in test_df_clean.columns:
                                        test_df_clean[col] = 0
                                
                                test_df_clean = test_df_clean[artifacts['feature_names']]
                                
                                # Scale
                                X_test_scaled = artifacts['scaler'].transform(test_df_clean)
                                
                                # Predict
                                predictions = artifacts['model'].predict(X_test_scaled)
                                pred_proba = artifacts['model'].predict_proba(X_test_scaled)
                                
                                # Decode predictions
                                if artifacts['label_encoder']:
                                    predictions_decoded = artifacts['label_encoder'].inverse_transform(predictions)
                                else:
                                    predictions_decoded = predictions
                                
                                # Display results
                                st.write("## 🎯 PREDICTION RESULTS")
                                
                                results_df = pd.DataFrame({
                                    'Sample_ID': range(1, len(predictions) + 1),
                                    'Predicted_Threat': predictions_decoded,
                                    'Confidence (%)': (pred_proba.max(axis=1) * 100).round(2)
                                })
                                
                                st.dataframe(results_df)
                                
                                # If labels exist, calculate metrics
                                if has_labels and y_true is not None:
                                    if artifacts['label_encoder']:
                                        # Check for unknown labels
                                        known_classes = set(artifacts['label_encoder'].classes_)
                                        test_classes = set(y_true.unique())
                                        unknown_classes = test_classes - known_classes
                                        
                                        if unknown_classes:
                                            st.warning(f"⚠️ Found unknown classes in test set: {unknown_classes}")
                                            st.info("Filtering out samples with unknown classes...")
                                            valid_mask = y_true.isin(known_classes)
                                            y_true = y_true[valid_mask]
                                            predictions = predictions[valid_mask.values]
                                            pred_proba = pred_proba[valid_mask.values]
                                            
                                            # Update results dataframe
                                            results_df = results_df[valid_mask.values].reset_index(drop=True)
                                            
                                            if len(y_true) == 0:
                                                st.error("❌ No valid labels remaining after filtering! Cannot evaluate model.")
                                                has_labels = False
                                        
                                        if has_labels:
                                            y_true_encoded = artifacts['label_encoder'].transform(y_true)
                                    else:
                                        y_true_encoded = y_true.values
                                    
                                    if has_labels:
                                        # Ensure inputs are numpy arrays for sklearn metric type expectations
                                        y_true_encoded = np.asarray(y_true_encoded)
                                        predictions = np.asarray(predictions)

                                        test_acc = accuracy_score(y_true_encoded, predictions) * 100
                                        test_prec = precision_score(y_true_encoded, predictions, average='weighted', zero_division=0) * 100
                                        test_rec = recall_score(y_true_encoded, predictions, average='weighted', zero_division=0) * 100
                                        test_f1 = f1_score(y_true_encoded, predictions, average='weighted', zero_division=0) * 100
                                        
                                        st.markdown(f"""
                                        <div class="success-box">
                                            <h1>🎯 TEST SET PERFORMANCE</h1>
                                            <h1 style="font-size: 5rem; margin: 1rem 0;">{test_acc:.2f}%</h1>
                                            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 2rem; margin-top: 2rem;">
                                                <div>
                                                    <h3>Precision</h3>
                                                    <h2 style="font-size: 2.5rem;">{test_prec:.2f}%</h2>
                                                </div>
                                                <div>
                                                    <h3>Recall</h3>
                                                    <h2 style="font-size: 2.5rem;">{test_rec:.2f}%</h2>
                                                </div>
                                                <div>
                                                    <h3>F1-Score</h3>
                                                    <h2 style="font-size: 2.5rem;">{test_f1:.2f}%</h2>
                                                </div>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Confusion Matrix
                                        st.write("### 🎯 Confusion Matrix")
                                        cm = confusion_matrix(y_true_encoded, predictions)
                                        
                                        fig, ax = plt.subplots(figsize=(10, 8))
                                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                                                   cbar_kws={'label': 'Count'}, linewidths=1, linecolor='gray')
                                        ax.set_title('Test Set Confusion Matrix', fontsize=16, fontweight='bold')
                                        ax.set_xlabel('Predicted Label', fontsize=12)
                                        ax.set_ylabel('Actual Label', fontsize=12)
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        
                                        # Classification Report
                                        st.write("### 📋 Detailed Classification Report")
                                        report = classification_report(y_true_encoded, predictions, output_dict=True, zero_division=0)
                                        report_df = pd.DataFrame(report).transpose().round(3)
                                        # removed width='stretch'
                                        st.dataframe(report_df)
                                
                                # Download predictions
                                csv = results_df.to_csv(index=False)
                                # removed width='stretch'
                                st.download_button(
                                    label="📥 Download Predictions CSV",
                                    data=csv,
                                    file_name="threat_predictions.csv",
                                    mime="text/csv"
                                )
                                
                            except Exception as e:
                                st.error(f"❌ Prediction Error: {str(e)}")
                                import traceback
                                with st.expander("🔍 View Error Details"):
                                    st.code(traceback.format_exc())
                
                except Exception as e:
                    st.error(f"❌ Error loading test file: {str(e)}")
    
    elif mode == "📊 View Results":
        st.write("## 📊 TRAINING RESULTS DASHBOARD")
         
        if st.session_state.test_results is None:
            st.markdown("""
            <div class="info-box">
                <h3>⚠️ No Results Available</h3>
                <p>Please train a model first to view results.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            try:
                results = st.session_state.test_results
                best_model_name = results.get('best_model_name', 'STACKING ENSEMBLE')
                
                # Main metrics
                best_acc = results['results'][best_model_name]['accuracy'] * 100
                best_prec = results['results'][best_model_name]['precision'] * 100
                best_rec = results['results'][best_model_name]['recall'] * 100
                best_f1 = results['results'][best_model_name]['f1_score'] * 100
                
                st.markdown(f"""
                <div class="success-box">
                    <h1>🏆 BEST MODEL: {best_model_name}</h1>
                    <h1 style="font-size: 6rem; margin: 2rem 0;">{best_acc:.2f}%</h1>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 3rem; margin-top: 3rem;">
                        <div style="background: rgba(255,255,255,0.2); padding: 1.5rem; border-radius: 15px;">
                            <h3 style="margin: 0;">Precision</h3>
                            <h1 style="font-size: 3rem; margin: 0.5rem 0;">{best_prec:.2f}%</h1>
                        </div>
                        <div style="background: rgba(255,255,255,0.2); padding: 1.5rem; border-radius: 15px;">
                            <h3 style="margin: 0;">Recall</h3>
                            <h1 style="font-size: 3rem; margin: 0.5rem 0;">{best_rec:.2f}%</h1>
                        </div>
                        <div style="background: rgba(255,255,255,0.2); padding: 1.5rem; border-radius: 15px;">
                            <h3 style="margin: 0;">F1-Score</h3>
                            <h1 style="font-size: 3rem; margin: 0.5rem 0;">{best_f1:.2f}%</h1>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # All models comparison
                st.write("### 📊 All Models Performance Comparison")
                results_df = pd.DataFrame({
                    'Model': list(results['results'].keys()),
                    'Accuracy (%)': [r['accuracy'] * 100 for r in results['results'].values()],
                    'Precision (%)': [r['precision'] * 100 for r in results['results'].values()],
                    'Recall (%)': [r['recall'] * 100 for r in results['results'].values()],
                    'F1-Score (%)': [r['f1_score'] * 100 for r in results['results'].values()]
                }).round(2)
                
                results_df = results_df.sort_values('Accuracy (%)', ascending=False)
                
                # removed width='stretch' (must be int or omitted)
                st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'))
                
                # Visualizations
                st.write("### 📈 Performance Visualizations")
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # Accuracy comparison
                axes[0, 0].barh(results_df['Model'], results_df['Accuracy (%)'], color='#11998e', edgecolor='black')
                axes[0, 0].set_xlabel('Accuracy (%)', fontweight='bold')
                axes[0, 0].set_title('Model Accuracy Comparison', fontweight='bold', fontsize=14)
                axes[0, 0].set_xlim(0, 100)
                axes[0, 0].grid(axis='x', alpha=0.3)
                
                # Precision comparison
                axes[0, 1].barh(results_df['Model'], results_df['Precision (%)'], color='#667eea', edgecolor='black')
                axes[0, 1].set_xlabel('Precision (%)', fontweight='bold')
                axes[0, 1].set_title('Model Precision Comparison', fontweight='bold', fontsize=14)
                axes[0, 1].set_xlim(0, 100)
                axes[0, 1].grid(axis='x', alpha=0.3)
                
                # Recall comparison
                axes[1, 0].barh(results_df['Model'], results_df['Recall (%)'], color='#38ef7d', edgecolor='black')
                axes[1, 0].set_xlabel('Recall (%)', fontweight='bold')
                axes[1, 0].set_title('Model Recall Comparison', fontweight='bold', fontsize=14)
                axes[1, 0].set_xlim(0, 100)
                axes[1, 0].grid(axis='x', alpha=0.3)
                
                # F1-Score comparison
                axes[1, 1].barh(results_df['Model'], results_df['F1-Score (%)'], color='#764ba2', edgecolor='black')
                axes[1, 1].set_xlabel('F1-Score (%)', fontweight='bold')
                axes[1, 1].set_title('Model F1-Score Comparison', fontweight='bold', fontsize=14)
                axes[1, 1].set_xlim(0, 100)
                axes[1, 1].grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Confusion Matrix
                st.write("### 🎯 Confusion Matrix")
                cm = confusion_matrix(results['y_test'], results['y_pred'])
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax, 
                           cbar_kws={'label': 'Count'}, linewidths=1, linecolor='black')
                ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=16, fontweight='bold')
                ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
                ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Classification Report
                st.write("### 📋 Detailed Classification Report")
                report = classification_report(results['y_test'], results['y_pred'], output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose().round(4)
                # removed width='stretch'
                st.dataframe(report_df)
                
                # Download buttons
                st.markdown("---")
                st.write("### 💾 Download Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_results = results_df.to_csv(index=False)
                    # removed width='stretch'
                    st.download_button(
                        label="📥 Download Model Comparison",
                        data=csv_results,
                        file_name="model_comparison.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    csv_report = report_df.to_csv()
                    st.download_button(
                        label="📥 Download Classification Report",
                        data=csv_report,
                        file_name="classification_report.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"❌ Error displaying results: {str(e)}")
                import traceback
                with st.expander("🔍 View Error Details"):
                    st.code(traceback.format_exc())
        
    elif mode == "📥 Dataset":
        st.markdown("""
            <div class="dataset-box">
                <h1>📊 Network Attack Detection Dataset</h1>
                <p style="font-size: 1.2rem;">Download and explore the dataset used to train this model. Contains sophisticated features extracted from network traffic for attack detection.</p>
            </div>
        """, unsafe_allow_html=True)
        
        try:
            # Full path to the dataset
            import os
            dataset_path = os.path.join(os.path.dirname(__file__), "network_attack_dataset.csv")
            
            # Dataset preview with file size info (memory-friendly)
            st.write("### 🔍 Dataset Preview")
            try:
                preview_df = read_csv_preview(dataset_path, nrows=10)
            except Exception:
                preview_df = None

            try:
                file_size = os.path.getsize(dataset_path) / (1024 * 1024)  # Convert to MB
                st.info(f"📦 File Size: {file_size:.2f} MB")
            except Exception:
                st.info("📦 File Size: N/A")

            if preview_df is not None:
                # removed width='stretch'
                st.dataframe(preview_df)
            else:
                st.info("Preview not available for this file.")

            # Dataset statistics (computed without loading full CSV)
            col1, col2, col3, col4 = st.columns(4)
            total_records = count_csv_lines(dataset_path)
            try:
                cols = pd.read_csv(dataset_path, nrows=0).columns
                num_features = max(len(cols) - 1, 0)
            except Exception:
                num_features = "N/A"

            label_counts = compute_label_counts(dataset_path)

            with col1:
                st.metric("📋 Total Records", f"{total_records:,}" if total_records is not None else "N/A")
            with col2:
                st.metric("📊 Features", f"{num_features}")
            with col3:
                if isinstance(label_counts, pd.Series) and 'attack' in label_counts.index:
                    st.metric("🎯 Attack Records", f"{int(label_counts['attack']):,}")
                else:
                    st.metric("🎯 Attack Records", "N/A")
            with col4:
                if isinstance(label_counts, pd.Series) and 'normal' in label_counts.index:
                    st.metric("✅ Normal Records", f"{int(label_counts['normal']):,}")
                else:
                    st.metric("✅ Normal Records", "N/A")
            
            # Download section with better error handling
            st.markdown("### 📥 Download Dataset")
            try:
                with open(dataset_path, "rb") as file:
                    csv_data = file.read()
                    st.download_button(
                        label="📥 Download Complete Dataset (CSV)",
                        data=csv_data,
                        file_name="network_attack_dataset.csv",
                        mime="text/csv"
                    )
                st.success("✅ Dataset is ready for download! Click the button above to start downloading.")
            except Exception as e:
                st.error(f"❌ Error preparing download: {str(e)}")
                st.info("💡 If you're having trouble downloading, please try accessing from a desktop browser or contact support.")
            
        except Exception as e:
            # Handle errors that occur while loading or reading the dataset file
            st.error(f"❌ Error loading dataset: {str(e)}")
            import traceback
            with st.expander("🔍 View Error Details"):
                st.code(traceback.format_exc())

        # Dataset documentation
        with st.expander("📋 View Complete Dataset Documentation"):
            doc_path = os.path.join(os.path.dirname(__file__), "Dataset discription", "dis.txt")
            doc_text = ""
            try:
                # Prefer utf-8, but fall back to common Windows encodings if necessary
                with open(doc_path, "r", encoding="utf-8") as doc_file:
                    doc_text = doc_file.read()
            except UnicodeDecodeError:
                try:
                    with open(doc_path, "r", encoding="cp1252") as doc_file:
                        doc_text = doc_file.read()
                except Exception:
                    try:
                        with open(doc_path, "r", encoding="latin-1") as doc_file:
                            doc_text = doc_file.read()
                    except Exception as e:
                        doc_text = f"❌ Error reading documentation file: {e}"
            except FileNotFoundError:
                doc_text = "⚠️ Documentation file not found."
            except Exception as e:
                doc_text = f"❌ Error reading documentation file: {e}"

            st.markdown(doc_text)
                
        # Feature descriptions
        st.markdown("### 🔰 Quick Feature Guide")
        st.markdown("""
        - **f1-f200**: Numerical features extracted from network traffic
        - **protocol_type**: Network protocol (tcp, udp, icmp)
        - **service**: Network service type
        - **flag**: Connection status flag
        - **label**: Classification (attack/normal)
        
        The dataset is pre-processed and ready for machine learning applications.
        Features are normalized and balanced for optimal training results.
        """)
        
        # Citation info
        st.markdown("### 📚 Citation & Usage")
        st.markdown("""
        If you use this dataset in your research or project, please cite:
        ```
        Network Attack Detection Dataset (2025)
        Cyber Threat Detection System
        ```
        """)
