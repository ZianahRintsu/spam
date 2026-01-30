import streamlit as st
import joblib
import re

# 1. Configuration de la page (Plus professionnel pour l'ISPM)
st.set_page_config(page_title="NLP Spam Detector - ISPM", page_icon="🛡️")

# --- CHARGEMENT DES ASSETS ---
@st.cache_resource
def load_assets():
    # Charge le modèle et le vectoriseur ré-entraînés avec les données FR
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_assets()
except Exception as e:
    st.error(f"Erreur de chargement des fichiers modèles : {e}")

# --- INTERFACE UTILISATEUR ---
st.title("🛡️ Détecteur de Spam Intelligent")
st.subheader("Projet NLP - Institut Supérieur Polytechnique de Madagascar")
st.write("Analyse bilingue (Français / Anglais) basée sur un modèle Random Forest et N-Grams.")

# Zone de saisie
message_input = st.text_area("Saisissez le SMS à analyser :", height=120, placeholder="Ex: Félicitations, vous avez gagné un lot...")

# Barre latérale pour les paramètres techniques (Valorise votre note)
st.sidebar.header("Paramètres du modèle")
threshold = st.sidebar.slider("Seuil de sensibilité", 0.1, 0.9, 0.5, help="Ajustez la sensibilité pour la détection du spam.")
st.sidebar.markdown("---")
st.sidebar.write("🌐 [www.ispm-edu.com](http://www.ispm-edu.com)")

if st.button("Lancer l'Analyse"):
    if message_input.strip() != "":
        # 2. PRÉTRAITEMENT (Identique à celui de l'entraînement)
        clean_text = message_input.lower()
        # On garde les accents pour le français
        clean_text = re.sub(r'[^a-z0-9àâçéèêëîïôûùÿ\s]', '', clean_text)
        
        # 3. PRÉDICTION VIA LE MODÈLE NATUREL
        vectorized_text = vectorizer.transform([clean_text])
        probabilities = model.predict_proba(vectorized_text)[0]
        spam_probability = probabilities[1]
        
        # 4. LOGIQUE DE DÉCISION
        is_spam = spam_probability >= threshold
        
        # 5. AFFICHAGE DES RÉSULTATS
                st.divider()
        if is_spam:
            st.error(f"🚨 **RÉSULTAT : SPAM**")
            st.warning(f"Confiance : {spam_probability*100:.2f}%")
        else:
            st.success(f"✅ **RÉSULTAT : HAM (Légitime)**")
            st.info(f"Confiance : {(1 - spam_probability)*100:.2f}%")
            
        # Barre de progression visuelle
        st.write("Probabilité de spam :")
        st.progress(spam_probability)
    else:
        st.warning("Veuillez entrer un message avant d'analyser.")

# Footer obligatoire pour l'examen
st.markdown("---")
st.caption("© 2026 - ISPM NLP Project - Master / Ingéniorat")
