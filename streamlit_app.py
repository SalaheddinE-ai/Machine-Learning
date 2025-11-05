import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
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
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
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
        
        # Bouton de prédiction
        predict_button = st.button('🔮 Prédire l\'espèce', type="primary", use_container_width=True)
    
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
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.success(f"### 🐧 Espèce prédite : **{predicted_species}**")
            st.info(f"**Confiance de la prédiction : {confidence:.2f}%**")
        
        with col2:
            # Emoji selon l'espèce
            species_emoji = {
                'Adelie': '🐧',
                'Chinstrap': '🐧',
                'Gentoo': '🐧'
            }
            st.markdown(f"<div style='font-size: 100px; text-align: center;'>{species_emoji[predicted_species]}</div>", 
                       unsafe_allow_html=True)
        
        # Graphique des probabilités
        st.subheader("📊 Probabilités par Espèce")
        
        df_proba = pd.DataFrame({
            'Espèce': species_names,
            'Probabilité': prediction_proba[0] * 100
        })
        
        fig = px.bar(
            df_proba, 
            x='Espèce', 
            y='Probabilité',
            color='Espèce',
            color_discrete_map={
                'Adelie': '#FF6B6B',
                'Chinstrap': '#4ECDC4',
                'Gentoo': '#45B7D1'
            },
            text='Probabilité'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(
            showlegend=False,
            yaxis_title="Probabilité (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau détaillé des probabilités
        with st.expander("📋 Tableau détaillé des probabilités"):
            df_proba_display = pd.DataFrame({
                'Espèce': species_names,
                'Probabilité (%)': [f"{p*100:.2f}%" for p in prediction_proba[0]],
                'Barre de progression': prediction_proba[0]
            })
            st.dataframe(
                df_proba_display,
                column_config={
                    'Barre de progression': st.column_config.ProgressColumn(
                        'Confiance',
                        min_value=0,
                        max_value=1,
                        format="%.2f"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
    
    # ========== TAB 2: DONNÉES ==========
    with tab2:
        st.header("📊 Exploration des Données")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📝 Nombre d'observations", len(df))
        with col2:
            st.metric("🔢 Nombre de variables", len(df.columns))
        with col3:
            st.metric("🐧 Nombre d'espèces", df['species'].nunique())
        with col4:
            st.metric("✅ Données complètes", "Oui")
        
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
        
        fig = px.pie(
            values=species_counts.values,
            names=species_counts.index,
            title="Répartition des Espèces dans le Dataset",
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ========== TAB 3: VISUALISATIONS ==========
    with tab3:
        st.header("📉 Visualisations des Données")
        
        # Scatter plot interactif
        st.subheader("🔍 Relation entre les Variables")
        
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox(
                'Axe X',
                ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'],
                index=0
            )
        with col2:
            y_axis = st.selectbox(
                'Axe Y',
                ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'],
                index=3
            )
        
        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color='species',
            symbol='sex',
            size='body_mass_g',
            hover_data=['island'],
            title=f'{y_axis} vs {x_axis}',
            color_discrete_map={
                'Adelie': '#FF6B6B',
                'Chinstrap': '#4ECDC4',
                'Gentoo': '#45B7D1'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Box plots
        st.subheader("📦 Distribution des Variables par Espèce")
        
        variable = st.selectbox(
            'Sélectionner une variable',
            ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
        )
        
        fig = px.box(
            df,
            x='species',
            y=variable,
            color='species',
            title=f'Distribution de {variable} par Espèce',
            color_discrete_map={
                'Adelie': '#FF6B6B',
                'Chinstrap': '#4ECDC4',
                'Gentoo': '#45B7D1'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("🔗 Matrice de Corrélation")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            title="Corrélations entre les Variables Numériques",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ========== TAB 4: PERFORMANCE ==========
    with tab4:
        st.header("🎯 Performance du Modèle")
        
        # Métriques de performance
        st.subheader("📊 Métriques Globales")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Précision du Modèle", f"{accuracy*100:.2f}%")
        with col2:
            st.metric("🌳 Nombre d'Arbres", n_estimators)
        with col3:
            st.metric("📏 Profondeur Max", max_depth)
        
        st.divider()
        
        # Matrice de confusion
        st.subheader("🔢 Matrice de Confusion")
        cm = confusion_matrix(y_test, y_pred)
        
        fig = px.imshow(
            cm,
            labels=dict(x="Prédiction", y="Valeur Réelle", color="Nombre"),
            x=species_names,
            y=species_names,
            text_auto=True,
            title="Matrice de Confusion",
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
        
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
        st.dataframe(report_df, use_container_width=True)
        
        # Feature importance
        st.subheader("🔍 Importance des Variables")
        
        feature_importance = pd.DataFrame({
            'Variable': X_encoded.columns,
            'Importance': clf.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Variable',
            orientation='h',
            title='Top 10 des Variables les Plus Importantes',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
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
        
        #### 🤖 Le Modèle
        - **Algorithme**: Random Forest Classifier
        - **Tâche**: Classification multi-classes
        - **Librairie**: scikit-learn
        
        #### 🎯 Caractéristiques de l'Application
        - ✅ Prédiction en temps réel
        - ✅ Visualisations interactives
        - ✅ Métriques de performance détaillées
        - ✅ Hyperparamètres ajustables
        - ✅ Interface intuitive
        
        #### 🛠️ Technologies Utilisées
        - **Streamlit**: Interface web
        - **Scikit-learn**: Machine Learning
        - **Plotly**: Visualisations interactives
        - **Pandas & NumPy**: Manipulation de données
        
        #### 📚 En savoir plus
        - [Dataset Palmer Penguins](https://github.com/allisonhorst/palmerpenguins)
        - [Documentation Streamlit](https://docs.streamlit.io)
        - [Documentation Scikit-learn](https://scikit-learn.org)
        
        ---
        
        Développé avec ❤️ pour l'apprentissage du Machine Learning
        """)
        
        # Informations système
        with st.expander("⚙️ Informations Système"):
            st.code(f"""
            Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            Python Libraries:
            - Streamlit: {st.__version__}
            - Pandas: {pd.__version__}
            - NumPy: {np.__version__}
            """)

else:
    st.error("❌ Impossible de charger les données. Veuillez vérifier votre connexion internet.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🐧 Application de Machine Learning - Prédiction d'Espèces de Manchots</p>
    <p>Créé avec Streamlit | © 2024</p>
</div>
""", unsafe_allow_html=True)
