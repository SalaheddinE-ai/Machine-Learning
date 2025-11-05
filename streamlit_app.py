import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime
import time

# Configuration de la page
st.set_page_config(
    page_title="Prédiction d'Espèces de Manchots",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .species-adelie { color: #FF6B6B; font-weight: bold; }
    .species-chinstrap { color: #4ECDC4; font-weight: bold; }
    .species-gentoo { color: #45B7D1; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<p class="main-header">🐧 Application de Machine Learning</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Prédiction d\'espèces de manchots avec Random Forest</p>', unsafe_allow_html=True)

# Fonction pour charger les données
@st.cache_data
def load_data():
    """Charge et prépare les données des manchots"""
    try:
        df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None

# Dictionnaire des modèles de Machine Learning
ML_MODELS = {
    'Random Forest': {
        'model': RandomForestClassifier,
        'params': {
            'n_estimators': {'type': 'slider', 'min': 10, 'max': 500, 'default': 100, 'step': 10, 'label': 'Nombre d\'arbres'},
            'max_depth': {'type': 'slider', 'min': 1, 'max': 30, 'default': 10, 'step': 1, 'label': 'Profondeur maximale'},
            'min_samples_split': {'type': 'slider', 'min': 2, 'max': 20, 'default': 2, 'step': 1, 'label': 'Min échantillons pour split'}
        },
        'description': '🌳 Ensemble d\'arbres de décision. Robuste et performant pour la classification.',
        'icon': '🌳'
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier,
        'params': {
            'n_estimators': {'type': 'slider', 'min': 10, 'max': 300, 'default': 100, 'step': 10, 'label': 'Nombre d\'estimateurs'},
            'learning_rate': {'type': 'slider', 'min': 0.01, 'max': 1.0, 'default': 0.1, 'step': 0.01, 'label': 'Taux d\'apprentissage'},
            'max_depth': {'type': 'slider', 'min': 1, 'max': 10, 'default': 3, 'step': 1, 'label': 'Profondeur maximale'}
        },
        'description': '🚀 Boosting séquentiel. Très performant mais plus lent à entraîner.',
        'icon': '🚀'
    },
    'Support Vector Machine': {
        'model': SVC,
        'params': {
            'C': {'type': 'slider', 'min': 0.01, 'max': 100.0, 'default': 1.0, 'step': 0.1, 'label': 'Paramètre C (régularisation)'},
            'kernel': {'type': 'selectbox', 'options': ['rbf', 'linear', 'poly', 'sigmoid'], 'default': 'rbf', 'label': 'Kernel'},
            'gamma': {'type': 'selectbox', 'options': ['scale', 'auto'], 'default': 'scale', 'label': 'Gamma'}
        },
        'description': '🎯 Machine à vecteurs de support. Excellent pour les données non-linéaires.',
        'icon': '🎯'
    },
    'K-Nearest Neighbors': {
        'model': KNeighborsClassifier,
        'params': {
            'n_neighbors': {'type': 'slider', 'min': 1, 'max': 20, 'default': 5, 'step': 1, 'label': 'Nombre de voisins'},
            'weights': {'type': 'selectbox', 'options': ['uniform', 'distance'], 'default': 'uniform', 'label': 'Poids'},
            'metric': {'type': 'selectbox', 'options': ['euclidean', 'manhattan', 'minkowski'], 'default': 'euclidean', 'label': 'Métrique'}
        },
        'description': '👥 Classification basée sur la proximité. Simple et intuitif.',
        'icon': '👥'
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier,
        'params': {
            'max_depth': {'type': 'slider', 'min': 1, 'max': 30, 'default': 10, 'step': 1, 'label': 'Profondeur maximale'},
            'min_samples_split': {'type': 'slider', 'min': 2, 'max': 20, 'default': 2, 'step': 1, 'label': 'Min échantillons pour split'},
            'criterion': {'type': 'selectbox', 'options': ['gini', 'entropy'], 'default': 'gini', 'label': 'Critère de division'}
        },
        'description': '🌲 Arbre de décision unique. Facile à interpréter et visualiser.',
        'icon': '🌲'
    },
    'Logistic Regression': {
        'model': LogisticRegression,
        'params': {
            'C': {'type': 'slider', 'min': 0.01, 'max': 100.0, 'default': 1.0, 'step': 0.1, 'label': 'Paramètre C (régularisation)'},
            'max_iter': {'type': 'slider', 'min': 100, 'max': 1000, 'default': 200, 'step': 100, 'label': 'Itérations maximales'},
            'solver': {'type': 'selectbox', 'options': ['lbfgs', 'liblinear', 'saga'], 'default': 'lbfgs', 'label': 'Solveur'}
        },
        'description': '📊 Régression logistique. Simple, rapide et efficace pour la classification linéaire.',
        'icon': '📊'
    },
    'Naive Bayes': {
        'model': GaussianNB,
        'params': {
            'var_smoothing': {'type': 'slider', 'min': 1e-12, 'max': 1e-5, 'default': 1e-9, 'step': 1e-11, 'label': 'Lissage de variance', 'format': '%.2e'}
        },
        'description': '🎲 Classificateur bayésien. Très rapide, idéal pour les grands datasets.',
        'icon': '🎲'
    },
    'Neural Network': {
        'model': MLPClassifier,
        'params': {
            'hidden_layer_sizes': {'type': 'selectbox', 'options': [(50,), (100,), (100, 50), (100, 100)], 'default': (100,), 'label': 'Architecture (couches cachées)'},
            'activation': {'type': 'selectbox', 'options': ['relu', 'tanh', 'logistic'], 'default': 'relu', 'label': 'Fonction d\'activation'},
            'learning_rate_init': {'type': 'slider', 'min': 0.0001, 'max': 0.1, 'default': 0.001, 'step': 0.0001, 'label': 'Taux d\'apprentissage', 'format': '%.4f'}
        },
        'description': '🧠 Réseau de neurones. Puissant pour les relations complexes.',
        'icon': '🧠'
    },
    'AdaBoost': {
        'model': AdaBoostClassifier,
        'params': {
            'n_estimators': {'type': 'slider', 'min': 10, 'max': 300, 'default': 50, 'step': 10, 'label': 'Nombre d\'estimateurs'},
            'learning_rate': {'type': 'slider', 'min': 0.01, 'max': 2.0, 'default': 1.0, 'step': 0.1, 'label': 'Taux d\'apprentissage'}
        },
        'description': '⚡ Adaptive Boosting. Combine des modèles faibles pour créer un modèle fort.',
        'icon': '⚡'
    }
}

# Fonction pour créer un modèle avec ses paramètres
def create_model(model_name, params):
    """Crée une instance du modèle avec les paramètres spécifiés"""
    model_class = ML_MODELS[model_name]['model']
    
    # Paramètres spéciaux pour certains modèles
    if model_name == 'Support Vector Machine':
        params['probability'] = True  # Nécessaire pour predict_proba
    elif model_name == 'Logistic Regression':
        params['multi_class'] = 'multinomial'
    elif model_name == 'Neural Network':
        params['max_iter'] = 500
        params['random_state'] = 42
    
    # Ajouter random_state si le modèle le supporte
    if model_name not in ['Naive Bayes', 'K-Nearest Neighbors']:
        params['random_state'] = 42
    
    return model_class(**params)

# Fonction pour l'encodage des données
def encode_data(df, encode_columns):
    """Encode les variables catégorielles"""
    return pd.get_dummies(df, columns=encode_columns, prefix=encode_columns)

# Fonction pour entraîner le modèle
@st.cache_resource
def train_model(X, y, model_name, model_params):
    """Entraîne le modèle sélectionné"""
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Créer et entraîner le modèle
    start_time = time.time()
    clf = create_model(model_name, model_params)
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Évaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    return clf, accuracy, X_test, y_test, y_pred, training_time, cv_mean, cv_std

# Chargement des données
df = load_data()

if df is not None:
    # Séparation X et y
    X_raw = df.drop('species', axis=1)
    y_raw = df['species']
    
    # Encodage de y
    target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
    y = y_raw.map(target_mapper)
    
    # Encodage de X
    encode_columns = ['island', 'sex']
    X_encoded = encode_data(X_raw, encode_columns)
    
    # ========== SIDEBAR - PARAMÈTRES ==========
    with st.sidebar:
        st.header("🤖 Sélection du Modèle")
        
        # Sélection du modèle
        model_name = st.selectbox(
            '🎯 Choisir un modèle de ML',
            options=list(ML_MODELS.keys()),
            help="Sélectionnez l'algorithme de Machine Learning à utiliser"
        )
        
        # Afficher la description du modèle
        st.info(f"{ML_MODELS[model_name]['icon']} {ML_MODELS[model_name]['description']}")
        
        st.divider()
        st.header("⚙️ Hyperparamètres")
        
        # Générer dynamiquement les contrôles pour les paramètres
        model_params = {}
        with st.expander("🔧 Ajuster les paramètres", expanded=True):
            for param_name, param_config in ML_MODELS[model_name]['params'].items():
                if param_config['type'] == 'slider':
                    format_str = param_config.get('format', None)
                    model_params[param_name] = st.slider(
                        param_config['label'],
                        min_value=param_config['min'],
                        max_value=param_config['max'],
                        value=param_config['default'],
                        step=param_config['step'],
                        format=format_str
                    )
                elif param_config['type'] == 'selectbox':
                    options = param_config['options']
                    default_idx = options.index(param_config['default']) if param_config['default'] in options else 0
                    model_params[param_name] = st.selectbox(
                        param_config['label'],
                        options=options,
                        index=default_idx
                    )
        
        st.divider()
        st.header("📊 Caractéristiques du Manchot")
        
        # Input features
        island = st.selectbox('🏝️ Île', ('Biscoe', 'Dream', 'Torgersen'))
        bill_length_mm = st.slider('📏 Longueur du bec (mm)', 32.1, 59.6, 43.9)
        bill_depth_mm = st.slider('📐 Profondeur du bec (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.slider('🦅 Longueur de la nageoire (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.slider('⚖️ Masse corporelle (g)', 2700.0, 6300.0, 4207.0)
        gender = st.selectbox('⚥ Sexe', ('male', 'female'))
        
        # Info
        st.divider()
        st.info("👆 Ajustez les paramètres ci-dessus puis consultez l'onglet **Prédiction**")
    
    # ========== ONGLETS PRINCIPAUX ==========
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Prédiction", 
        "📊 Données", 
        "📉 Visualisations", 
        "🎯 Performance du Modèle",
        "⚖️ Comparaison des Modèles",
        "ℹ️ À propos"
    ])
    
    # ========== TAB 1: PRÉDICTION ==========
    with tab1:
        st.header("🔮 Prédiction de l'Espèce")
        
        # Créer le DataFrame d'entrée
        input_data = {
            'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': gender
        }
        input_df = pd.DataFrame(input_data, index=[0])
        
        # Afficher les caractéristiques saisies
        st.subheader("📝 Caractéristiques saisies")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🏝️ Île", island)
            st.metric("📏 Longueur du bec", f"{bill_length_mm} mm")
        with col2:
            st.metric("⚥ Sexe", gender.capitalize())
            st.metric("📐 Profondeur du bec", f"{bill_depth_mm} mm")
        with col3:
            st.metric("🦅 Longueur nageoire", f"{flipper_length_mm} mm")
            st.metric("⚖️ Masse corporelle", f"{body_mass_g} g")
        
        # Entraîner le modèle
        with st.spinner(f'🤖 Entraînement du modèle {model_name} en cours...'):
            clf, accuracy, X_test, y_test, y_pred, training_time, cv_mean, cv_std = train_model(
                X_encoded, y, model_name, model_params
            )
        
        # Préparer les données pour la prédiction
        input_penguins = pd.concat([input_df, X_raw], axis=0)
        input_encoded = encode_data(input_penguins, encode_columns)
        input_row = input_encoded[:1]
        
        # Assurer que toutes les colonnes sont présentes
        missing_cols = set(X_encoded.columns) - set(input_row.columns)
        for col in missing_cols:
            input_row[col] = 0
        input_row = input_row[X_encoded.columns]
        
        # Faire la prédiction
        prediction = clf.predict(input_row)
        prediction_proba = clf.predict_proba(input_row)
        
        # Résultats de prédiction
        st.divider()
        st.subheader("🎯 Résultats de la Prédiction")
        
        # Informations sur le modèle utilisé
        st.info(f"**Modèle utilisé**: {ML_MODELS[model_name]['icon']} {model_name}")
        
        species_names = ['Adelie', 'Chinstrap', 'Gentoo']
        predicted_species = species_names[prediction[0]]
        confidence = prediction_proba[0][prediction[0]] * 100
        
        # Affichage de la prédiction principale
        col1, col2 = st.columns([3, 1])
        
        with col1:
            species_class = f"species-{predicted_species.lower()}"
            st.markdown(f"""
            <div class="prediction-box">
                <h2>🐧 Espèce prédite</h2>
                <h1 style="font-size: 3rem; margin: 1rem 0;">{predicted_species}</h1>
                <h3>Confiance : {confidence:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"<div style='font-size: 120px; text-align: center; margin-top: 2rem;'>🐧</div>", 
                       unsafe_allow_html=True)
        
        # Tableau des probabilités
        st.subheader("📊 Probabilités par Espèce")
        
        df_proba = pd.DataFrame({
            'Espèce': species_names,
            'Probabilité (%)': [f"{p*100:.2f}%" for p in prediction_proba[0]],
            'Confiance': prediction_proba[0]
        })
        
        st.dataframe(
            df_proba,
            column_config={
                'Espèce': st.column_config.TextColumn('🐧 Espèce', width='medium'),
                'Probabilité (%)': st.column_config.TextColumn('📊 Probabilité', width='medium'),
                'Confiance': st.column_config.ProgressColumn(
                    '📈 Niveau de confiance',
                    min_value=0,
                    max_value=1,
                    format="%.2f"
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Graphique avec bar_chart natif de Streamlit
        st.subheader("📈 Graphique de Probabilités")
        chart_data = pd.DataFrame({
            'Probabilité': prediction_proba[0] * 100
        }, index=species_names)
        st.bar_chart(chart_data)
    
    # ========== TAB 2: DONNÉES ==========
    with tab2:
        st.header("📊 Exploration des Données")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Observations", len(df))
        with col2:
            st.metric("🔢 Variables", len(df.columns))
        with col3:
            st.metric("🐧 Espèces", df['species'].nunique())
        with col4:
            st.metric("✅ Complétude", "100%")
        
        st.divider()
        
        # Aperçu des données
        st.subheader("👀 Aperçu des Données Brutes")
        st.dataframe(df, use_container_width=True, height=400)
        
        # Statistiques descriptives
        with st.expander("📈 Statistiques Descriptives"):
            st.dataframe(df.describe(), use_container_width=True)
        
        # Distribution des espèces
        st.subheader("📊 Distribution des Espèces")
        species_counts = df['species'].value_counts()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.bar_chart(species_counts)
        with col2:
            st.dataframe(
                pd.DataFrame({
                    'Espèce': species_counts.index,
                    'Nombre': species_counts.values,
                    'Pourcentage': [f"{(v/len(df)*100):.1f}%" for v in species_counts.values]
                }),
                hide_index=True
            )
        
        # Informations par espèce
        st.subheader("📋 Résumé par Espèce")
        for species in df['species'].unique():
            with st.expander(f"🐧 {species}"):
                species_df = df[df['species'] == species]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nombre", len(species_df))
                with col2:
                    st.metric("Masse moyenne", f"{species_df['body_mass_g'].mean():.0f} g")
                with col3:
                    st.metric("Bec moyen", f"{species_df['bill_length_mm'].mean():.1f} mm")
    
    # ========== TAB 3: VISUALISATIONS ==========
    with tab3:
        st.header("📉 Visualisations des Données")
        
        # Scatter plot avec Streamlit
        st.subheader("🔍 Relation entre les Variables")
        
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox(
                'Axe X',
                ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'],
                index=0,
                key='x_axis'
            )
        with col2:
            y_axis = st.selectbox(
                'Axe Y',
                ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'],
                index=3,
                key='y_axis'
            )
        
        # Scatter chart natif Streamlit
        st.scatter_chart(
            data=df,
            x=x_axis,
            y=y_axis,
            color='species',
            size='body_mass_g'
        )
        
        # Distribution par variable
        st.subheader("📊 Distribution des Variables")
        
        variable = st.selectbox(
            'Sélectionner une variable à analyser',
            ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'],
            key='dist_var'
        )
        
        st.write(f"**Distribution de {variable.replace('_', ' ').title()}**")
        
        # Histogramme pour chaque espèce
        for species in df['species'].unique():
            species_data = df[df['species'] == species][variable]
            st.write(f"**{species}**: Moyenne = {species_data.mean():.2f}, Écart-type = {species_data.std():.2f}")
        
        # Comparaison avec line chart
        comparison_df = df.groupby('species')[variable].mean().reset_index()
        comparison_df.columns = ['Espèce', 'Valeur Moyenne']
        st.bar_chart(comparison_df.set_index('Espèce'))
        
        # Matrice de corrélation simplifiée
        st.subheader("🔗 Corrélations entre Variables")
        numeric_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
        corr_matrix = df[numeric_cols].corr().round(3)
        
        st.dataframe(
            corr_matrix,
            use_container_width=True,
            column_config={col: st.column_config.NumberColumn(
                col.replace('_', ' ').title(),
                format="%.3f"
            ) for col in corr_matrix.columns}
        )
        
        st.caption("💡 Les valeurs proches de 1 indiquent une forte corrélation positive, proche de -1 une forte corrélation négative, et proche de 0 aucune corrélation.")
    
    # ========== TAB 4: PERFORMANCE ==========
    with tab4:
        st.header("🎯 Performance du Modèle")
        
        # Métriques de performance
        st.subheader("📊 Métriques Globales")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Précision", f"{accuracy*100:.2f}%", 
                     delta=f"{(accuracy-0.8)*100:.1f}%" if accuracy > 0.8 else None)
        with col2:
            st.metric("🌳 Arbres", model_params.get('n_estimators', 'N/A'))
        with col3:
            st.metric("📏 Profondeur", model_params.get('max_depth', 'N/A'))
        with col4:
            st.metric("⚙️ Temps d'entraînement", f"{training_time:.2f} sec")
        
        st.divider()
        
        # Matrice de confusion
        st.subheader("🔢 Matrice de Confusion")
        cm = confusion_matrix(y_test, y_pred)
        
        cm_df = pd.DataFrame(
            cm,
            index=[f'Réel: {s}' for s in species_names],
            columns=[f'Prédit: {s}' for s in species_names]
        )
        
        st.dataframe(
            cm_df,
            use_container_width=True
        )
        
        # Rapport de classification
        st.subheader("📋 Rapport de Classification Détaillé")
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=species_names,
            output_dict=True
        )
        
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.round(3)
        st.dataframe(
            report_df,
            use_container_width=True
        )
        
        # Feature importance (si disponible)
        st.subheader("🔍 Importance des Variables")
        
        # Vérifier si le modèle supporte feature_importances_
        if hasattr(clf, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Variable': X_encoded.columns,
                'Importance': clf.feature_importances_
            }).sort_values('Importance', ascending=False).head(15)
            
            st.dataframe(
                feature_importance,
                column_config={
                    'Variable': st.column_config.TextColumn('📊 Variable', width='large'),
                    'Importance': st.column_config.ProgressColumn(
                        '📈 Importance',
                        min_value=0,
                        max_value=feature_importance['Importance'].max(),
                        format="%.4f"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Graphique d'importance
            st.bar_chart(feature_importance.set_index('Variable')['Importance'])
        elif hasattr(clf, 'coef_'):
            st.info("📊 Ce modèle utilise des coefficients au lieu d'importance de variables")
            # Afficher les coefficients pour les modèles linéaires
            coef_abs = np.abs(clf.coef_).mean(axis=0)
            feature_coef = pd.DataFrame({
                'Variable': X_encoded.columns,
                'Coefficient (abs)': coef_abs
            }).sort_values('Coefficient (abs)', ascending=False).head(15)
            
            st.dataframe(feature_coef, hide_index=True, use_container_width=True)
            st.bar_chart(feature_coef.set_index('Variable')['Coefficient (abs)'])
        else:
            st.warning("⚠️ Ce modèle ne fournit pas d'information sur l'importance des variables")
    
    # ========== TAB 5: COMPARAISON DES MODÈLES ==========
    with tab5:
        st.header("📊 Comparaison des Modèles")
        
        st.info("🔄 Cette section compare les performances de tous les modèles disponibles")
        
        if st.button("🚀 Lancer la comparaison des modèles", type="primary"):
            comparison_results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (m_name, m_config) in enumerate(ML_MODELS.items()):
                status_text.text(f"Entraînement de {m_name}...")
                
                # Utiliser les paramètres par défaut
                default_params = {}
                for param_name, param_config in m_config['params'].items():
                    default_params[param_name] = param_config['default']
                
                try:
                    # Entraîner le modèle
                    start = time.time()
                    model = create_model(m_name, default_params)
                    
                    X_train, X_test_temp, y_train, y_test_temp = train_test_split(
                        X_encoded, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    model.fit(X_train, y_train)
                    train_time = time.time() - start
                    
                    # Évaluer
                    y_pred_temp = model.predict(X_test_temp)
                    acc = accuracy_score(y_test_temp, y_pred_temp)
                    
                    # Cross-validation
                    cv_scores_temp = cross_val_score(model, X_train, y_train, cv=5)
                    
                    comparison_results.append({
                        'Modèle': m_name,
                        'Icon': m_config['icon'],
                        'Précision Test': f"{acc*100:.2f}%",
                        'Précision CV': f"{cv_scores_temp.mean()*100:.2f}%",
                        'CV Std': f"±{cv_scores_temp.std()*100:.2f}%",
                        'Temps (s)': f"{train_time:.3f}",
                        'Score': acc
                    })
                except Exception as e:
                    st.warning(f"⚠️ Erreur avec {m_name}: {str(e)}")
                
                progress_bar.progress((idx + 1) / len(ML_MODELS))
            
            status_text.text("✅ Comparaison terminée!")
            
            # Afficher les résultats
            if comparison_results:
                st.subheader("🏆 Résultats de la Comparaison")
                
                comparison_df = pd.DataFrame(comparison_results)
                comparison_df = comparison_df.sort_values('Score', ascending=False)
                comparison_df = comparison_df.drop('Score', axis=1)
                
                # Ajouter des médailles
                if len(comparison_df) >= 3:
                    comparison_df['Rang'] = ['🥇', '🥈', '🥉'] + [''] * (len(comparison_df) - 3)
                    comparison_df = comparison_df[['Rang', 'Icon', 'Modèle', 'Précision Test', 'Précision CV', 'CV Std', 'Temps (s)']]
                
                st.dataframe(comparison_df, hide_index=True, use_container_width=True)
                
                # Conseils
                st.success(f"🏆 **Meilleur modèle**: {comparison_df.iloc[0]['Modèle']} avec une précision de {comparison_df.iloc[0]['Précision Test']}")
                
                st.divider()
                st.subheader("💡 Conseils pour choisir un modèle")
                st.markdown("""
                - **Précision élevée** : Choisissez le modèle avec la meilleure précision test
                - **Rapidité** : Si le temps est important, privilégiez les modèles rapides (Naive Bayes, Logistic Regression)
                - **Interprétabilité** : Decision Tree et Logistic Regression sont plus faciles à interpréter
                - **Robustesse** : Random Forest et Gradient Boosting sont généralement plus robustes
                - **Données complexes** : Neural Network pour les relations non-linéaires complexes
                """)
        else:
            st.write("👆 Cliquez sur le bouton ci-dessus pour comparer tous les modèles")
            
            # Tableau récapitulatif des modèles
            st.subheader("📋 Modèles Disponibles")
            models_info = []
            for m_name, m_config in ML_MODELS.items():
                models_info.append({
                    'Icon': m_config['icon'],
                    'Modèle': m_name,
                    'Description': m_config['description']
                })
            
            models_df = pd.DataFrame(models_info)
            st.dataframe(models_df, hide_index=True, use_container_width=True)
    
    # ========== TAB 6: À PROPOS ==========
    with tab6:
        st.header("ℹ️ À propos de l'Application")
        
        st.markdown("""
        ### 🐧 Dataset des Manchots de Palmer
        
        Cette application utilise le célèbre dataset des manchots de Palmer pour démontrer 
        les capacités du Machine Learning dans la classification d'espèces.
        
        #### 📊 Les Données
        - **Source**: Palmer Station, Antarctique
        - **Espèces**: Adelie, Chinstrap, et Gentoo
        - **Variables**: Mesures morphologiques des manchots
        - **Îles**: Biscoe, Dream, et Torgersen
        
        #### 🤖 Les Modèles Disponibles
        
        Cette application propose **9 algorithmes de Machine Learning** différents:
        
        1. **🌳 Random Forest**: Ensemble d'arbres de décision
        2. **🚀 Gradient Boosting**: Boosting séquentiel puissant
        3. **🎯 SVM**: Machine à vecteurs de support
        4. **👥 K-Nearest Neighbors**: Classification par proximité
        5. **🌲 Decision Tree**: Arbre de décision simple
        6. **📊 Logistic Regression**: Classification linéaire
        7. **🎲 Naive Bayes**: Classificateur bayésien
        8. **🧠 Neural Network**: Réseau de neurones multicouche
        9. **⚡ AdaBoost**: Adaptive Boosting
        
        #### 🎯 Caractéristiques de l'Application
        - ✅ **9 modèles de ML** au choix
        - ✅ **Hyperparamètres personnalisables** pour chaque modèle
        - ✅ **Prédiction en temps réel**
        - ✅ **Comparaison automatique** des modèles
        - ✅ **Validation croisée** (5-fold CV)
        - ✅ **Métriques détaillées** (précision, matrice de confusion, rapport de classification)
        - ✅ **Visualisations interactives** natives
        - ✅ **Interface intuitive** et responsive
        
        #### 🛠️ Technologies Utilisées
        - **Streamlit**: Framework d'interface web
        - **Scikit-learn**: Bibliothèque de Machine Learning
        - **Pandas**: Manipulation et analyse de données
        - **NumPy**: Calculs numériques
        
        #### 📈 Comment utiliser l'application
        
        1. **Sidebar**: 
           - Sélectionnez un modèle de ML
           - Ajustez ses hyperparamètres
           - Définissez les caractéristiques du manchot
        2. **Onglet Prédiction**: Visualisez la prédiction du modèle
        3. **Onglet Données**: Explorez le dataset
        4. **Onglet Visualisations**: Analysez les relations entre variables
        5. **Onglet Performance**: Évaluez la qualité du modèle
        6. **Onglet Comparaison**: Comparez tous les modèles automatiquement
        
        #### 🏆 Conseils pour de Meilleures Prédictions
        
        - **Random Forest** et **Gradient Boosting** offrent généralement les meilleures performances
        - **SVM** est excellent pour les données non-linéaires
        - **Logistic Regression** est rapide et simple pour débuter
        - **Neural Network** peut capturer des relations complexes mais nécessite plus de données
        - Utilisez l'**onglet Comparaison** pour trouver le meilleur modèle
        
        #### 📚 Ressources
        - [Dataset Palmer Penguins](https://github.com/allisonhorst/palmerpenguins)
        - [Documentation Streamlit](https://docs.streamlit.io)
        - [Documentation Scikit-learn](https://scikit-learn.org)
        - [Guide des algorithmes de classification](https://scikit-learn.org/stable/supervised_learning.html)
        
        #### 🔬 Variables du Dataset
        
        | Variable | Description | Unité |
        |----------|-------------|-------|
        | island | Île où le manchot a été observé | Catégorielle |
        | bill_length_mm | Longueur du bec | mm |
        | bill_depth_mm | Profondeur du bec | mm |
        | flipper_length_mm | Longueur de la nageoire | mm |
        | body_mass_g | Masse corporelle | g |
        | sex | Sexe du manchot | Catégorielle |
        | species | Espèce (cible) | Catégorielle |
        
        ---
        
        💡 **Astuce**: Utilisez l'onglet "Comparaison des Modèles" pour identifier automatiquement 
        le meilleur algorithme pour ce dataset!
        
        ---
        
        Développé avec ❤️ pour l'apprentissage du Machine Learning
        """)
        
        # Informations système
        with st.expander("⚙️ Informations Système"):
            st.code(f"""
📅 Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

📚 Python Libraries:
- Streamlit: {st.__version__}
- Pandas: {pd.__version__}
- NumPy: {np.__version__}

🤖 Modèle Actuel: {model_name}
🎯 Précision: {accuracy*100:.2f}%
✅ Précision CV: {cv_mean*100:.2f}% (±{cv_std*100:.2f}%)
⏱️ Temps d'entraînement: {training_time:.3f}s

📊 Dataset:
- Observations: {len(df)}
- Variables: {len(df.columns)}
- Espèces: {df['species'].nunique()}

🔧 Paramètres:
{chr(10).join([f'- {k}: {v}' for k, v in model_params.items()])}
            """)
        
        # Exemples de prédiction
        with st.expander("🎮 Exemples de Configuration"):
            st.markdown("""
            **Exemple 1 - Manchot Adelie typique:**
            - Île: Torgersen
            - Longueur du bec: 39 mm
            - Profondeur du bec: 18 mm
            - Longueur nageoire: 190 mm
            - Masse: 3700 g
            
            **Exemple 2 - Manchot Gentoo typique:**
            - Île: Biscoe
            - Longueur du bec: 47 mm
            - Profondeur du bec: 15 mm
            - Longueur nageoire: 217 mm
            - Masse: 5000 g
            
            **Exemple 3 - Manchot Chinstrap typique:**
            - Île: Dream
            - Longueur du bec: 49 mm
            - Profondeur du bec: 18 mm
            - Longueur nageoire: 195 mm
            - Masse: 3800 g
            """)

else:
    st.error("❌ Impossible de charger les données. Veuillez vérifier votre connexion internet.")
    st.info("💡 Assurez-vous d'avoir une connexion internet active pour charger le dataset.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🐧 <strong>Application de Machine Learning</strong> - Prédiction d'Espèces de Manchots</p>
    <p>Créé avec Streamlit 🎈 | © 2024</p>
</div>
""", unsafe_allow_html=True)
