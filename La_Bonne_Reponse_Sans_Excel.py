# import streamlit as st
# import google.generativeai as genai
# import PIL.Image
# from datetime import datetime
# from dotenv import load_dotenv
# import os

# load_dotenv()

# # Configuration de la page
# st.set_page_config(
#     page_title="Analyseur de Plans d'Architecture",
#     page_icon="🏗️",
#     layout="wide"
# )

# # Configuration de l'API
# @st.cache_resource
# def configure_genai():
#     genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#     return genai.GenerativeModel('gemini-2.5-flash')

# # Fonction pour redimensionner l'image
# def resize_image_if_needed(image, max_size=1024):
#     """Redimensionne l'image si elle est trop grande pour accélérer le traitement"""
#     width, height = image.size
#     if max(width, height) > max_size:
#         # Calculer le ratio pour maintenir les proportions
#         ratio = max_size / max(width, height)
#         new_width = int(width * ratio)
#         new_height = int(height * ratio)
#         return image.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
#     return image

# # Interface principale
# def main():
#     st.title("🏗️ Analyseur de Plans d'Architecture")
#     st.markdown("*Analyse automatique avec IA - Détection d'objets et calcul de surfaces*")
#     st.markdown("---")
    
#     # Sidebar pour les paramètres
#     with st.sidebar:
#         st.header("⚙️ Configuration")
        
#         # Mode d'analyse
#         st.subheader("⚡ Mode d'analyse")
#         analysis_mode = st.radio(
#             "Choisissez le mode:",
#             ("🚀 Rapide", "🔍 Détaillé"),
#             help="Mode rapide: analyse basique en 10-15s. Mode détaillé: analyse complète en 30-60s"
#         )
        
#         # Prompt personnalisé
#         st.subheader("📝 Personnalisation du prompt")
#         custom_prompt = st.text_area(
#             "Prompt personnalisé (optionnel)",
#             placeholder="Ajoutez des instructions spécifiques...",
#             height=100
#         )
        
#         st.markdown("---")
#         st.markdown("**💡 Astuce:** Uploadez un plan clair et bien éclairé pour de meilleurs résultats.")
    
#     # Corps principal
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.header("📤 Upload du Plan")
        
#         uploaded_file = st.file_uploader(
#             "Choisissez un fichier image",
#             type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
#             help="Formats supportés: JPG, JPEG, PNG, BMP, TIFF"
#         )
        
#         if uploaded_file is not None:
#             # Afficher l'image
#             image = PIL.Image.open(uploaded_file)
            
#             # Optimisation : redimensionner si l'image est trop grande
#             original_size = image.size
#             image = resize_image_if_needed(image, max_size=1024)
            
#             st.image(image, caption="Plan d'architecture uploadé", width=None)
            
#             if original_size != image.size:
#                 st.info(f"🔧 Image redimensionnée de {original_size} à {image.size} pour accélérer l'analyse")
            
#             # Informations sur l'image
#             with st.expander("ℹ️ Informations sur l'image"):
#                 st.write(f"**Nom:** {uploaded_file.name}")
#                 st.write(f"**Dimensions:** {image.size[0]} x {image.size[1]} pixels")
#                 st.write(f"**Taille:** {round(uploaded_file.size / 1024, 2)} KB")
#                 st.write(f"**Format:** {image.format}")
    
#     with col2:
#         st.header("🔍 Analyse et Résultats")
        
#         if uploaded_file is not None:
#             if st.button("🚀 Analyser le Plan", type="primary"):
#                 with st.spinner("🤖 Analyse en cours avec l'IA..."):
#                     try:
#                         # Configurer le modèle
#                         model = configure_genai()
                        
#                         # Construire le prompt selon le mode choisi
#                         if analysis_mode == "🚀 Rapide":
#                             base_prompt = """
# Analysez rapidement ce plan d'architecture et fournissez :

# 1. **Liste des pièces** avec leur surface approximative en m²
# 2. **Nombre total de portes et fenêtres**
# 3. **Équipements visibles** (cuisine, salle de bain, etc.)
# 4. **Observations générales** sur le plan

# Organisez votre réponse de manière claire et structurée.
#                             """.strip()
#                         else:
#                             base_prompt = """
# Analysez ce plan d'architecture de manière détaillée et fournissez :

# ## 🏠 PIÈCES ET ESPACES
# Pour chaque pièce, indiquez :
# - Nom et localisation
# - Surface approximative (en m²)
# - Dimensions approximatives (longueur x largeur x hauteur si visible)
# - Équipements et aménagements présents

# ## 🚪 OUVERTURES
# - Nombre et dimensions des portes
# - Nombre et dimensions des fenêtres
# - Types d'ouvertures (standard, coulissante, etc.)

# ## 🔌 ÉQUIPEMENTS TECHNIQUES
# - Prises électriques et leurs emplacements
# - Équipements électriques (tableau, VMC, etc.)
# - Équipements sanitaires et de cuisine
# - Autres équipements techniques visibles

# ## 🏗️ STRUCTURE ET MURS
# - Description des murs (porteurs, cloisons)
# - Escaliers (type, localisation, dimensions)
# - Éléments structurels remarquables

# ## 📊 RÉSUMÉ QUANTITATIF
# - Surface totale approximative
# - Nombre total de pièces
# - Nombre total de portes et fenêtres

# Organisez votre réponse de manière claire avec des titres et sous-titres.
#                             """.strip()
                        
#                         if custom_prompt:
#                             base_prompt += f"\n\n**Instructions supplémentaires :** {custom_prompt}"
                        
#                         # Analyse avec Gemini
#                         try:
#                             response = model.generate_content([base_prompt, image])
                            
#                             # Stocker les résultats
#                             st.session_state.analysis_result = response.text
#                             st.session_state.analysis_timestamp = datetime.now()
                            
#                             st.success("✅ Analyse terminée avec succès!")
                            
#                         except Exception as e:
#                             st.error(f"⏱️ Timeout ou erreur: {str(e)}")
#                             st.info("💡 Essayez le mode rapide ou une image plus petite")
                        
#                     except Exception as e:
#                         st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
    
#     # Affichage des résultats
#     if 'analysis_result' in st.session_state:
#         st.markdown("---")
#         st.header("📊 Résultats de l'Analyse")
        
#         # Onglets pour les différentes vues
#         tab1, tab2 = st.tabs([
#             "📝 Analyse complète", 
#             "💾 Téléchargement"
#         ])
        
#         with tab1:
#             st.subheader("📝 Rapport d'Analyse Détaillé")
            
#             # Afficher les résultats avec un meilleur formatage
#             st.markdown(st.session_state.analysis_result)
        
#         with tab2:
#             st.subheader("💾 Télécharger les Résultats")
#             st.write("Téléchargez le rapport d'analyse :")
            
#             # Téléchargement du rapport
#             st.download_button(
#                 label="📄 Télécharger le Rapport (TXT)",
#                 data=st.session_state.analysis_result,
#                 file_name=f"analyse_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
#                 mime="text/plain",
#                 help="Rapport d'analyse complet au format texte"
#             )
            
#             # Informations sur l'analyse
#             st.markdown("---")
#             st.info(f"🕒 Analyse effectuée le: {st.session_state.analysis_timestamp.strftime('%d/%m/%Y à %H:%M:%S')}")
    
#     # Footer
#     st.markdown("---")
#     st.markdown(
#         """
#         <div style='text-align: center; color: gray; font-size: 0.9em;'>
#             🏗️ Analyseur de Plans d'Architecture - Powered by Google Gemini AI<br>
#             Développé avec Streamlit • Version 2.3 (Simplifié)
#         </div>
#         """, 
#         unsafe_allow_html=True
#     )

# if __name__ == "__main__":
#     main()

import streamlit as st
import google.generativeai as genai
import PIL.Image
from datetime import datetime
from dotenv import load_dotenv
import os
import io

# Charger les variables d'environnement
load_dotenv()

# Configuration de la page
st.set_page_config(
    page_title="Analyseur de Plans d'Architecture",
    page_icon="🏗️",
    layout="wide"
)

# Fonction pour configurer le modèle Gemini
def configure_genai():
    st.write("🔧 Configuration du modèle Gemini...")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-2.5-flash')
    st.write("✅ Modèle configuré avec succès")
    return model

# Fonction pour redimensionner l'image
def resize_image_if_needed(image, max_size=1024):
    width, height = image.size
    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return image.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
    return image

# Convertir l'image PIL en bytes pour Gemini
def image_to_bytes(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return buffered.getvalue()

# Interface principale
def main():
    st.title("🏗️ Analyseur de Plans d'Architecture")
    st.markdown("*Analyse automatique avec IA - Détection d'objets et calcul de surfaces*")
    st.markdown("---")

    # Sidebar pour les paramètres
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.subheader("⚡ Mode d'analyse")
        analysis_mode = st.radio(
            "Choisissez le mode:",
            ("🚀 Rapide", "🔍 Détaillé"),
            help="Mode rapide: analyse basique en 10-15s. Mode détaillé: analyse complète en 30-60s"
        )

        st.subheader("📝 Personnalisation du prompt")
        custom_prompt = st.text_area(
            "Prompt personnalisé (optionnel)",
            placeholder="Ajoutez des instructions spécifiques...",
            height=100
        )

        st.markdown("---")
        st.markdown("**💡 Astuce:** Uploadez un plan clair et bien éclairé pour de meilleurs résultats.")

    # Corps principal
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload du Plan")
        uploaded_file = st.file_uploader(
            "Choisissez un fichier image",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Formats supportés: JPG, JPEG, PNG, BMP, TIFF"
        )

        if uploaded_file is not None:
            image = PIL.Image.open(uploaded_file)
            original_size = image.size
            image = resize_image_if_needed(image, max_size=1024)
            st.image(image, caption="Plan d'architecture uploadé", width=None)
            if original_size != image.size:
                st.info(f"🔧 Image redimensionnée de {original_size} à {image.size} pour accélérer l'analyse")

            with st.expander("ℹ️ Informations sur l'image"):
                st.write(f"**Nom:** {uploaded_file.name}")
                st.write(f"**Dimensions:** {image.size[0]} x {image.size[1]} pixels")
                st.write(f"**Taille:** {round(uploaded_file.size / 1024, 2)} KB")
                st.write(f"**Format:** {image.format}")

    with col2:
        st.header("🔍 Analyse et Résultats")

        if uploaded_file is not None:
            if st.button("🚀 Analyser le Plan", type="primary"):
                with st.spinner("🤖 Analyse en cours avec l'IA..."):
                    try:
                        st.write("📌 Étape 1: Configuration du modèle")
                        model = configure_genai()

                        st.write("📌 Étape 2: Construction du prompt")
                        if analysis_mode == "🚀 Rapide":
                            base_prompt = """
Analysez rapidement ce plan d'architecture et fournissez :

1. **Liste des pièces** avec leur surface approximative en m²
2. **Nombre total de portes et fenêtres**
3. **Équipements visibles** (cuisine, salle de bain, etc.)
4. **Observations générales** sur le plan

Organisez votre réponse de manière claire et structurée.
                            """.strip()
                        else:
                            base_prompt = """
Analysez ce plan d'architecture de manière détaillée et fournissez :

## 🏠 PIÈCES ET ESPACES
Pour chaque pièce, indiquez :
- Nom et localisation
- Surface approximative (en m²)
- Dimensions approximatives (longueur x largeur x hauteur si visible)
- Équipements et aménagements présents

## 🚪 OUVERTURES
- Nombre et dimensions des portes
- Nombre et dimensions des fenêtres
- Types d'ouvertures (standard, coulissante, etc.)

## 🔌 ÉQUIPEMENTS TECHNIQUES
- Prises électriques et leurs emplacements
- Équipements électriques (tableau, VMC, etc.)
- Équipements sanitaires et de cuisine
- Autres équipements techniques visibles

## 🏗️ STRUCTURE ET MURS
- Description des murs (porteurs, cloisons)
- Escaliers (type, localisation, dimensions)
- Éléments structurels remarquables

## 📊 RÉSUMÉ QUANTITATIF
- Surface totale approximative
- Nombre total de pièces
- Nombre total de portes et fenêtres

Organisez votre réponse de manière claire avec des titres et sous-titres.
                            """.strip()

                        if custom_prompt:
                            base_prompt += f"\n\n**Instructions supplémentaires :** {custom_prompt}"

                        st.write("📌 Étape 3: Conversion de l'image en bytes")
                        

                        st.write("📌 Étape 4: Appel au modèle Gemini")
                        response = model.generate_content([base_prompt, image])

                        st.write("📌 Étape 5: Réponse reçue du modèle")

                        st.session_state.analysis_result = response.text
                        st.session_state.analysis_timestamp = datetime.now()
                        st.success("✅ Analyse terminée avec succès!")

                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
                        st.info("💡 Essayez le mode rapide ou une image plus petite")

    # Affichage des résultats
    if 'analysis_result' in st.session_state:
        st.markdown("---")
        st.header("📊 Résultats de l'Analyse")
        tab1, tab2 = st.tabs(["📝 Analyse complète", "💾 Téléchargement"])

        with tab1:
            st.subheader("📝 Rapport d'Analyse Détaillé")
            st.markdown(st.session_state.analysis_result)

        with tab2:
            st.subheader("💾 Télécharger les Résultats")
            st.download_button(
                label="📄 Télécharger le Rapport (TXT)",
                data=st.session_state.analysis_result,
                file_name=f"analyse_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                help="Rapport d'analyse complet au format texte"
            )
            st.markdown("---")
            st.info(f"🕒 Analyse effectuée le: {st.session_state.analysis_timestamp.strftime('%d/%m/%Y à %H:%M:%S')}")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.9em;'>"
        "🏗️ Analyseur de Plans d'Architecture - Powered by Google Gemini AI<br>"
        "Développé avec Streamlit • Version 2.3 (Simplifié)"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
