import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime

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

# Fonction pour l'encodage des données
def encode_data(df, encode_columns):
    """Encode les variables catégorielles"""
    return pd.get_dummies(df, columns=encode_columns, prefix=encode_columns)

# Fonction pour entraîner le modèle
@st.cache_resource
def train_model(X, y, n_estimators=100, max_depth=None, random_state=42):
    """Entraîne le modèle Random Forest"""
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Entraînement du modèle
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Évaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return clf, accuracy, X_test, y_test, y_pred

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
        st.header("⚙️ Configuration du Modèle")
        
        # Hyperparamètres
        with st.expander("🔧 Hyperparamètres", expanded=False):
            n_estimators = st.slider('Nombre d\'arbres', 10, 500, 100, 10)
            max_depth = st.slider('Profondeur maximale', 1, 20, 10)
            random_state = st.number_input('Random State', 0, 100, 42)
        
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Prédiction", 
        "📊 Données", 
        "📉 Visualisations", 
        "🎯 Performance du Modèle",
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
        with st.spinner('🤖 Entraînement du modèle en cours...'):
            clf, accuracy, X_test, y_test, y_pred = train_model(
                X_encoded, y, n_estimators, max_depth, random_state
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
        corr_matrix = df[numeric_cols].corr()
        st.dataframe(
            corr_matrix.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1),
            use_container_width=True
        )
    
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
            st.metric("🌳 Arbres", n_estimators)
        with col3:
            st.metric("📏 Profondeur", max_depth)
        with col4:
            st.metric("🔢 Random State", random_state)
        
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
            cm_df.style.background_gradient(cmap='Blues'),
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
            report_df.style.background_gradient(cmap='Greens', subset=['precision', 'recall', 'f1-score']),
            use_container_width=True
        )
        
        # Feature importance
        st.subheader("🔍 Importance des Variables")
        
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
    
    # ========== TAB 5: À PROPOS ==========
    with tab5:
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
        
        #### 🤖 Le Modèle
        - **Algorithme**: Random Forest Classifier
        - **Tâche**: Classification multi-classes (3 espèces)
        - **Librairie**: scikit-learn
        - **Entraînement**: 80% des données
        - **Test**: 20% des données
        
        #### 🎯 Caractéristiques de l'Application
        - ✅ Prédiction en temps réel
        - ✅ Visualisations interactives natives
        - ✅ Métriques de performance détaillées
        - ✅ Hyperparamètres ajustables
        - ✅ Interface intuitive et responsive
        - ✅ Cache intelligent pour les performances
        
        #### 🛠️ Technologies Utilisées
        - **Streamlit**: Framework d'interface web
        - **Scikit-learn**: Bibliothèque de Machine Learning
        - **Pandas**: Manipulation et analyse de données
        - **NumPy**: Calculs numériques
        
        #### 📈 Comment utiliser l'application
        
        1. **Sidebar**: Ajustez les caractéristiques du manchot
        2. **Onglet Prédiction**: Visualisez la prédiction du modèle
        3. **Onglet Données**: Explorez le dataset
        4. **Onglet Visualisations**: Analysez les relations entre variables
        5. **Onglet Performance**: Évaluez la qualité du modèle
        
        #### 📚 Ressources
        - [Dataset Palmer Penguins](https://github.com/allisonhorst/palmerpenguins)
        - [Documentation Streamlit](https://docs.streamlit.io)
        - [Documentation Scikit-learn](https://scikit-learn.org)
        - [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
        
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
        
        💡 **Conseil**: Essayez différentes combinaisons de paramètres pour voir comment 
        le modèle réagit et améliore ses prédictions!
        
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

🎯 Configuration du Modèle:
- Arbres: {n_estimators}
- Profondeur max: {max_depth}
- Random state: {random_state}
- Précision: {accuracy*100:.2f}%

📊 Dataset:
- Observations: {len(df)}
- Variables: {len(df.columns)}
- Espèces: {df['species'].nunique()}
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
