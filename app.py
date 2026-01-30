import streamlit as st
import joblib
import re

# Configuration de la page
st.set_page_config(page_title="Détecteur de Spam - ISPM", page_icon="🚫")

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource # Pour éviter de recharger le modèle à chaque clic
def load_assets():
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_assets()
except:
    st.error("Erreur : Les fichiers modèles (.pkl) sont introuvables.")

# --- INTERFACE UTILISATEUR ---
st.title("🛡️ Détecteur de SMS Spam")
st.markdown("### Institut Supérieur Polytechnique de Madagascar")
st.write("Entrez un message ci-dessous pour analyser s'il s'agit d'un message légitime (HAM) ou d'une arnaque (SPAM).")

# Zone de saisie
message_input = st.text_area("Saisissez votre SMS :", height=150, placeholder="Ex: Félicitations! Vous avez gagné...")

# Seuil de décision configurable (Bonus demandé !)
threshold = st.sidebar.slider("Seuil de détection (Sensibilité)", 0.0, 1.0, 0.5)

if st.button("Analyser le message"):
    if message_input.strip() != "":
        # 1. Prétraitement simple (identique à l'entraînement)
        clean_text = message_input.lower()
        clean_text = re.sub(r'[^a-z0-9\s]', '', clean_text)

        #TEST INJECTION DICTIONNAIRE FR ET MLG
        # Liste de mots suspects en FR et MG
        french_spam_keywords = ["gagné", "félicitations", "loka", "antsoy", "cadeau", "urgent", "cliquez", "lotery"]

        # Vérification manuelle (Bonus : Robustesse)
        is_manual_spam = any(word in message_input.lower() for word in french_spam_keywords)

        # Prédiction IA
        vectorized_text = vectorizer.transform([clean_text])
        probabilities = model.predict_proba(vectorized_text)[0]
        spam_probability = probabilities[1]

        # Si un mot clé est trouvé, on booste la probabilité
        if is_manual_spam:
            spam_probability = max(spam_probability, 0.85)

    #FIN DU TEST
        
        # 2. Vectorisation
    #    vectorized_text = vectorizer.transform([clean_text])
        
        # 3. Prédiction des probabilités
    #    probabilities = model.predict_proba(vectorized_text)[0]
    #   spam_probability = probabilities[1] # Probabilité de la classe 1 (Spam)
        
        # 4. Application du seuil
        is_spam = spam_probability >= threshold
        
        # 5. Affichage du résultat
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

st.sidebar.info("Projet NLP - ISPM 2026")