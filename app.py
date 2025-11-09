import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
    """Create advanced engineered features for 90%+ accuracy"""
    try:
        st.write("### ⚡ ADVANCED FEATURE ENGINEERING")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Select top features
            k_best = min(20, len(numeric_cols))
            selector = SelectKBest(mutual_info_classif, k=k_best)
            selector.fit(X[numeric_cols], y)
            top_features = [numeric_cols[i] for i in selector.get_support(indices=True)]
            
            st.write(f"🎯 Selected top {len(top_features)} features")
            
            feature_count = 0
            
            # Interaction features
            for i in range(min(8, len(top_features))):
                for j in range(i+1, min(i+5, len(top_features))):
                    col1, col2 = top_features[i], top_features[j]
                    
                    # Multiplication
                    X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                    
                    # Safe division
                    X[f'{col1}_div_{col2}'] = X[col1] / (np.abs(X[col2]) + 1e-8)
                    
                    # Addition & Subtraction
                    X[f'{col1}_plus_{col2}'] = X[col1] + X[col2]
                    X[f'{col1}_minus_{col2}'] = X[col1] - X[col2]
                    
                    feature_count += 4
            
            # Polynomial features
            for col in top_features[:10]:
                X[f'{col}_sq'] = X[col] ** 2
                X[f'{col}_sqrt'] = np.sqrt(np.abs(X[col]))
                X[f'{col}_log'] = np.log1p(np.abs(X[col]))
                feature_count += 3
            
            # Statistical aggregations
            X['row_mean'] = X[top_features].mean(axis=1)
            X['row_std'] = X[top_features].std(axis=1)
            X['row_max'] = X[top_features].max(axis=1)
            X['row_min'] = X[top_features].min(axis=1)
            X['row_range'] = X['row_max'] - X['row_min']
            feature_count += 5
            
            st.write(f"✅ Created {feature_count} engineered features")
        
        return X
        
    except Exception as e:
        st.warning(f"⚠️ Feature engineering warning: {str(e)}")
        return X


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
            email = st.text_input("📧 Email", value="user@example.com")
            password = st.text_input("🔑 Password", type="password", value="password123")
            login = st.form_submit_button("🚀 Login", use_container_width=True)
            
            if login:
                if email == "user@example.com" and password == "password123":
                    st.session_state.authenticated = True
                    st.success("✅ Login Successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Use: user@example.com / password123")
else:
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    mode = st.sidebar.radio("Select Mode", ["📚 Train Model", "🧪 Test Model", "📊 View Results", "📥 Dataset"], index=0)
    
    if mode == "📚 Train Model":
        st.sidebar.markdown("---")
        st.sidebar.header("📁 Upload Dataset")
        uploaded_file = st.sidebar.file_uploader("Upload Training CSV", type="csv", help="Upload your labeled training dataset")
        
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
            min_value=300,
            max_value=1500,
            value=800,
            step=100,
            help="More trees = better accuracy but slower training"
        )
        
        if uploaded_file:
            try:
                # Load data
                df = pd.read_csv(uploaded_file)
                
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
                    st.dataframe(df.head(30), use_container_width=True)
                
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
                    st.dataframe(class_counts, use_container_width=True)
                    
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
                if st.button("🚀 START TRAINING", use_container_width=True):
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
                            
                            # Step 7: Scale features
                            st.write("⚖️ Scaling features with RobustScaler + StandardScaler...")
                            scaler1 = RobustScaler()
                            scaler2 = StandardScaler()
                            
                            X_train_scaled = scaler1.fit_transform(X_train)
                            X_train_scaled = scaler2.fit_transform(X_train_scaled)
                            
                            X_test_scaled = scaler1.transform(X_test)
                            X_test_scaled = scaler2.transform(X_test_scaled)
                            
                            # Step 8: Balance classes
                            st.write(f"⚖️ Balancing classes using {balance_method}...")
                            
                            if balance_method == "SMOTETomek":
                                balancer = SMOTETomek(random_state=42, n_jobs=-1)
                            elif balance_method == "SMOTEENN":
                                balancer = SMOTEENN(random_state=42, n_jobs=-1)
                            elif balance_method == "BorderlineSMOTE":
                                balancer = BorderlineSMOTE(random_state=42, n_jobs=-1)
                            else:
                                balancer = SMOTE(random_state=42, n_jobs=-1)
                            
                            X_train_bal, y_train_bal = balancer.fit_resample(X_train_scaled, y_train)
                            st.write(f"✅ Balanced: {len(y_train):,} → {len(y_train_bal):,} samples")
                            
                            # Step 9: Train models
                            st.write("## 🚀 TRAINING ENSEMBLE MODELS")
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            models = {
                                'Random Forest': RandomForestClassifier(
                                    n_estimators=n_estimators,
                                    max_depth=None,
                                    min_samples_split=2,
                                    min_samples_leaf=1,
                                    max_features='sqrt',
                                    bootstrap=True,
                                    class_weight='balanced_subsample',
                                    criterion='gini',
                                    random_state=42,
                                    n_jobs=-1
                                ),
                                'Extra Trees': ExtraTreesClassifier(
                                    n_estimators=n_estimators,
                                    max_depth=None,
                                    min_samples_split=2,
                                    min_samples_leaf=1,
                                    max_features='sqrt',
                                    bootstrap=True,
                                    class_weight='balanced_subsample',
                                    criterion='gini',
                                    random_state=42,
                                    n_jobs=-1
                                ),
                                'Gradient Boosting': GradientBoostingClassifier(
                                    n_estimators=min(500, n_estimators),
                                    learning_rate=0.05,
                                    max_depth=12,
                                    min_samples_split=2,
                                    subsample=0.85,
                                    max_features='sqrt',
                                    random_state=42
                                ),
                                'AdaBoost': AdaBoostClassifier(
                                    n_estimators=min(300, n_estimators),
                                    learning_rate=0.8,
                                    algorithm='SAMME',
                                    random_state=42
                                ),
                                'Bagging Ensemble': BaggingClassifier(
                                    estimator=RandomForestClassifier(
                                        n_estimators=100,
                                        max_depth=25,
                                        class_weight='balanced',
                                        random_state=42
                                    ),
                                    n_estimators=50,
                                    max_samples=0.8,
                                    max_features=0.8,
                                    bootstrap=True,
                                    random_state=42,
                                    n_jobs=-1
                                )
                            }
                            
                            results = {}
                            trained_models = []
                            total_models = len(models) + 2  # +2 for voting and stacking
                            
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
                                weights=[2, 2, 1.5, 1, 1],
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
                                'scaler1': scaler1,
                                'scaler2': scaler2,
                                'drop_cols': drop_cols,
                                'cat_cols': cat_cols,
                                'label_encoder': le,
                                'feature_names': X.columns.tolist()
                            }
                            
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
                            
                            st.dataframe(
                                results_df.style.highlight_max(axis=0, color='lightgreen'),
                                use_container_width=True
                            )
                            
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
                                    <p style="font-size: 1.1rem; margin-top: 1rem;">Below 90% target. Consider these improvements:</p>
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
                                if st.button("💾 SAVE MODEL", use_container_width=True):
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
            
            test_file = st.file_uploader("📁 Upload Test Dataset (CSV)", type="csv")
            
            if test_file:
                try:
                    test_df = pd.read_csv(test_file)
                    
                    st.markdown(f"""
                    <div class="metric-box">
                        <h3>📊 Test Dataset Loaded</h3>
                        <p style="font-size: 1.5rem;">{test_df.shape[0]:,} rows × {test_df.shape[1]} columns</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("🔍 Preview Test Data (First 20 Rows)"):
                        st.dataframe(test_df.head(20), use_container_width=True)
                    
                    if st.button("🚀 RUN PREDICTION", use_container_width=True):
                        with st.spinner("🔄 Making predictions..."):
                            try:
                                artifacts = st.session_state.model_artifacts
                                
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
                                X_test_scaled = artifacts['scaler1'].transform(test_df_clean)
                                X_test_scaled = artifacts['scaler2'].transform(X_test_scaled)
                                
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
                                
                                st.dataframe(results_df, use_container_width=True)
                                
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
                                        st.dataframe(report_df, use_container_width=True)
                                
                                # Download predictions
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Predictions CSV",
                                    data=csv,
                                    file_name="threat_predictions.csv",
                                    mime="text/csv",
                                    use_container_width=True
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
        
    elif mode == "📥 Dataset":
        st.markdown("""
            <div class="dataset-box">
                <h1>📊 Network Attack Detection Dataset</h1>
                <p style="font-size: 1.2rem;">Download and explore the dataset used to train this model. Contains sophisticated features extracted from network traffic for attack detection.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Dataset preview
        st.write("### 🔍 Dataset Preview")
        df = pd.read_csv("network_attack_dataset.csv")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Dataset statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📋 Total Records", f"{len(df):,}")
        with col2:
            st.metric("📊 Features", f"{len(df.columns)-1}")
        with col3:
            st.metric("🎯 Attack Records", f"{len(df[df['label']=='attack']):,}")
        with col4:
            st.metric("✅ Normal Records", f"{len(df[df['label']=='normal']):,}")
        
        # Download section
        st.markdown("### 📥 Download Dataset")
        with open("network_attack_dataset.csv", "rb") as file:
            st.download_button(
                label="📥 Download Complete Dataset (CSV)",
                data=file,
                file_name="network_attack_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        # Dataset documentation
        with st.expander("📋 View Complete Dataset Documentation"):
            with open("Dataset discription/dis.txt", "r") as doc_file:
                st.markdown(doc_file.read())
                
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
                
                st.dataframe(
                    results_df.style.highlight_max(axis=0, color='lightgreen'),
                    use_container_width=True
                )
                
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
                st.dataframe(report_df, use_container_width=True)
                
                # Download buttons
                st.markdown("---")
                st.write("### 💾 Download Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_results = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Model Comparison",
                        data=csv_results,
                        file_name="model_comparison.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    csv_report = report_df.to_csv()
                    st.download_button(
                        label="📥 Download Classification Report",
                        data=csv_report,
                        file_name="classification_report.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
            except Exception as e:
                st.error(f"❌ Error displaying results: {str(e)}")
                import traceback
                with st.expander("🔍 View Error Details"):
                    st.code(traceback.format_exc())
    
    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Quick Info")
    st.sidebar.info("""
    **Tips for 90%+ Accuracy:**
    - Use clean, labeled data
    - Enable feature engineering
    - Use 800-1500 estimators
    - Try SMOTETomek balancing
    - Use 10-15% test split
    """)
    
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.trained_model = None
        st.session_state.test_results = None
        st.session_state.model_artifacts = None
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🛡️ <strong>Cyber Threat Detection System v2.0</strong></p>
    <p>Built with Streamlit | Powered by Ensemble Machine Learning</p>
</div>
""", unsafe_allow_html=True)
