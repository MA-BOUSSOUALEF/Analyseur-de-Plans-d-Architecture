import streamlit as st
import google.generativeai as genai
import PIL.Image
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration de la page
st.set_page_config(
    page_title="Analyseur de Plans d'Architecture",
    page_icon="🏗️",
    layout="wide"
)

# Configuration de l'API
@st.cache_resource
def configure_genai():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    return genai.GenerativeModel('gemini-2.5-flash')

# Fonction pour redimensionner l'image
def resize_image_if_needed(image, max_size=1024):
    """Redimensionne l'image si elle est trop grande pour accélérer le traitement"""
    width, height = image.size
    if max(width, height) > max_size:
        # Calculer le ratio pour maintenir les proportions
        ratio = max_size / max(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return image.resize((new_width, new_height), PIL.Image.Resampling.LANCZOS)
    return image

# Interface principale
def main():
    st.title("🏗️ Analyseur de Plans d'Architecture")
    st.markdown("*Analyse automatique avec IA - Détection d'objets et calcul de surfaces*")
    st.markdown("---")
    
    # Sidebar pour les paramètres
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode d'analyse
        st.subheader("⚡ Mode d'analyse")
        analysis_mode = st.radio(
            "Choisissez le mode:",
            ("🚀 Rapide", "🔍 Détaillé"),
            help="Mode rapide: analyse basique en 10-15s. Mode détaillé: analyse complète en 30-60s"
        )
        
        # Prompt personnalisé
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
            # Afficher l'image
            image = PIL.Image.open(uploaded_file)
            
            # Optimisation : redimensionner si l'image est trop grande
            original_size = image.size
            image = resize_image_if_needed(image, max_size=1024)
            
            st.image(image, caption="Plan d'architecture uploadé", width=None)
            
            if original_size != image.size:
                st.info(f"🔧 Image redimensionnée de {original_size} à {image.size} pour accélérer l'analyse")
            
            # Informations sur l'image
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
                        # Configurer le modèle
                        model = configure_genai()
                        
                        # Construire le prompt selon le mode choisi
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
                        
                        # Analyse avec Gemini
                        try:
                            response = model.generate_content([base_prompt, image])
                            
                            # Stocker les résultats
                            st.session_state.analysis_result = response.text
                            st.session_state.analysis_timestamp = datetime.now()
                            
                            st.success("✅ Analyse terminée avec succès!")
                            
                        except Exception as e:
                            st.error(f"⏱️ Timeout ou erreur: {str(e)}")
                            st.info("💡 Essayez le mode rapide ou une image plus petite")
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
    
    # Affichage des résultats
    if 'analysis_result' in st.session_state:
        st.markdown("---")
        st.header("📊 Résultats de l'Analyse")
        
        # Onglets pour les différentes vues
        tab1, tab2 = st.tabs([
            "📝 Analyse complète", 
            "💾 Téléchargement"
        ])
        
        with tab1:
            st.subheader("📝 Rapport d'Analyse Détaillé")
            
            # Afficher les résultats avec un meilleur formatage
            st.markdown(st.session_state.analysis_result)
        
        with tab2:
            st.subheader("💾 Télécharger les Résultats")
            st.write("Téléchargez le rapport d'analyse :")
            
            # Téléchargement du rapport
            st.download_button(
                label="📄 Télécharger le Rapport (TXT)",
                data=st.session_state.analysis_result,
                file_name=f"analyse_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                help="Rapport d'analyse complet au format texte"
            )
            
            # Informations sur l'analyse
            st.markdown("---")
            st.info(f"🕒 Analyse effectuée le: {st.session_state.analysis_timestamp.strftime('%d/%m/%Y à %H:%M:%S')}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.9em;'>
            🏗️ Analyseur de Plans d'Architecture - Powered by Google Gemini AI<br>
            Développé avec Streamlit • Version 2.3 (Simplifié)
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

# import streamlit as st
# import google.generativeai as genai
# import PIL.Image
# import json
# from datetime import datetime
# from dotenv import load_dotenv
# import os
# import re

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
#     return genai.GenerativeModel('gemini-2.5-flash')  # Plus rapide que 2.5-flash

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

# # Fonction pour extraire les données structurées de l'analyse
# def parse_analysis_text(text):
#     """Parse le texte d'analyse pour extraire les données selon le format JSON"""
#     data = {
#         "pieces_detaillees": [],
#         "escaliers": [],
#         "murs": [],
#         "elements_techniques_globaux": {},
#         "texte_complet": text
#     }
    
#     # Extraire le JSON s'il existe
#     json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
#     json_data = None
#     if json_match:
#         try:
#             json_data = json.loads(json_match.group(1))
#         except:
#             pass
    
#     # Essayer de parser le JSON directement si pas de balises ```json
#     if not json_data:
#         try:
#             json_data = json.loads(text.strip())
#         except:
#             pass
    
#     # Parser selon le format JSON
#     if json_data:
#         # Extraire les pièces avec le format
#         if "pieces" in json_data:
#             for piece in json_data["pieces"]:
#                 # Extraire les dimensions
#                 dimensions = piece.get("dimensions_m", {})
#                 largeur = dimensions.get("largeur", "") if dimensions else ""
#                 longueur = dimensions.get("longueur", "") if dimensions else ""
#                 hauteur = dimensions.get("hauteur", "") if dimensions else ""
                
#                 # Compter et extraire les portes
#                 portes = piece.get("portes", [])
#                 nb_portes = len(portes) if portes else 0
#                 dim_portes = " | ".join([p.get("dimension_m", "") for p in portes if p.get("dimension_m", "")])
                
#                 # Compter et extraire les fenêtres
#                 fenetres = piece.get("fenetres", [])
#                 nb_fenetres = len(fenetres) if fenetres else 0
#                 dim_fenetres = " | ".join([f.get("dimension_m", "") for f in fenetres if f.get("dimension_m", "")])
                
#                 # Extraire les équipements électriques
#                 equipements_electriques = piece.get("equipements_electriques", [])
#                 nb_prises_electriques = len(equipements_electriques) if equipements_electriques else 0
#                 dim_prises_electriques = " | ".join([e.get("dimension_m", "") for e in equipements_electriques if e.get("dimension_m", "")])
#                 equipements_electriques_text = " | ".join([f"{e.get('nom', '')} ({e.get('dimension_m', '')})" for e in equipements_electriques])
                
#                 # Extraire les autres équipements
#                 autres_equipements = piece.get("autres_equipements", [])
#                 autres_equipements_text = " | ".join([f"{e.get('nom', '')} ({e.get('dimension_m', '')})" for e in autres_equipements])
                
#                 # Combiner tous les équipements
#                 equipements_total = []
#                 if equipements_electriques_text:
#                     equipements_total.append(f"Électriques: {equipements_electriques_text}")
#                 if autres_equipements_text:
#                     equipements_total.append(f"Autres: {autres_equipements_text}")
                
#                 piece_info = {
#                     "piece": piece.get("nom", ""),
#                     "localisation": piece.get("localisation", ""),
#                     "surface_m2": piece.get("surface_m2", ""),
#                     "largeur_m": largeur,
#                     "longueur_m": longueur,
#                     "hauteur_m": hauteur,
#                     "equipements": " | ".join(equipements_total),
#                     "nb_portes": nb_portes,
#                     "nb_fenetres": nb_fenetres,
#                     "nb_prises_electriques": nb_prises_electriques,
#                     "dim_portes": dim_portes,
#                     "dim_fenetres": dim_fenetres,
#                     "dim_prises_electriques": dim_prises_electriques
#                 }
                
#                 data["pieces_detaillees"].append(piece_info)
        
#         # Extraire les informations sur les murs
#         if "murs" in json_data:
#             murs_info = json_data["murs"]
#             data["murs"].append({
#                 "localisation": murs_info.get("localisation", ""),
#                 "largeur": murs_info.get("largeur", ""),
#                 "longueur": murs_info.get("longueur", ""),
#                 "hauteur": murs_info.get("hauteur", ""),
#                 "description": murs_info.get("description", "")
#             })
        
#         # Extraire les informations sur l'escalier
#         if "escalier" in json_data:
#             escalier_info = json_data["escalier"]
#             data["escaliers"].append({
#                 "type": escalier_info.get("type", ""),
#                 "localisation": escalier_info.get("localisation", ""),
#                 "dimensions": escalier_info.get("dimensions_m", ""),
#                 "direction": escalier_info.get("direction", "")
#             })
        
#         # Extraire les totaux
#         data["elements_techniques_globaux"]["total_portes"] = json_data.get("total_portes", "")
#         data["elements_techniques_globaux"]["total_fenetres"] = json_data.get("total_fenetres", "")
    
#     else:
#         # Parsing basique si pas de JSON (fallback)
#         piece_pattern = r'(\w+(?:\s+\w+)*?):\s*(\d+[,\.]\d+)m²'
#         pieces_matches = re.findall(piece_pattern, text, re.IGNORECASE)
        
#         for nom, surface in pieces_matches:
#             nom_clean = nom.strip().upper()
#             surface_float = float(surface.replace(',', '.'))
#             data["pieces_detaillees"].append({
#                 "piece": nom_clean,
#                 "localisation": "",
#                 "surface_m2": surface_float,
#                 "largeur_m": "",
#                 "longueur_m": "",
#                 "hauteur_m": "",
#                 "equipements": "",
#                 "nb_portes": "",
#                 "nb_fenetres": "",
#                 "nb_prises_electriques": "",
#                 "dim_portes": "",
#                 "dim_fenetres": "",
#                 "dim_prises_electriques": ""
#             })
    
#     return data

# # Interface principale
# def main():
#     st.title("🏗️ Analyseur de Plans d'Architecture")
#     st.markdown("*Analyse automatique avec IA - Détection d'objets et calcul de surfaces*")
#     st.markdown("---")
    
#             # Sidebar pour les paramètres
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
        
#         # Options d'analyse
#         st.subheader("🔍 Options d'analyse")
#         analyze_surfaces = st.checkbox("Calculer les surfaces", value=True)
#         analyze_equipment = st.checkbox("Détecter les équipements", value=True)
#         analyze_technical = st.checkbox("Compter les éléments techniques", value=True)
#         json_output = st.checkbox("Sortie JSON structurée", value=True)
        
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
# Analysez rapidement ce plan d'architecture (réponse en 20 secondes maximum).

# Listez simplement :
# 1. Les pièces principales avec leur surface approximative en m²
# 2. Le nombre total de portes et fenêtres
# 3. Les équipements visibles (cuisine, salle de bain, etc.)

# Répondez au format JSON simple :
# {
#   "pieces": [
#     {"nom": "", "surface_m2": "" ,"largeur_m": "", "longueur_m": "", "hauteur_m": "","equipements":""}
#   ],
#   "total_portes": "",
#   "total_fenetres": "",
#   "equipements_principaux": []
# }
#                             """.strip()
#                         else:
#                             base_prompt = """
# Analysez ce plan d'architecture et fournissez les informations suivantes :

# Listez toutes les pièces avec :
# - leur nom
# - localisation
# - surface (en m²)
# - dimensions approximatives (longueur, largeur, hauteur si disponible)
# - portes avec leurs dimensions
# - fenêtres avec leurs dimensions
# - équipements électriques avec dimensions si visibles (prises, VRE, etc.)
# - autres équipements avec leurs dimensions (mobilier, électroménager, sanitaires…)

# Décrivez les murs :
# - localisation
# - largeur
# - longueur
# - hauteur
# - description (type intérieur/extérieur, porteur ou cloison, etc.)

# Décrivez l'escalier :
# - type
# - localisation
# - dimensions approximatives
# - direction

# Comptez le nombre de portes et fenêtres total

# Répondez au format JSON suivant :
# {
#   "pieces": [
#     {
#       "nom": "",
#       "localisation": "",
#       "surface_m2": "",
#       "dimensions_m": {
#         "longueur": "",
#         "largeur": "",
#         "hauteur": ""
#       },
#       "portes": [
#         { "dimension_m": "" }
#       ],
#       "fenetres": [
#         { "dimension_m": "" }
#       ],
#       "tous_les_equipements": [
#         { "nom": "", "dimension_m": "" }
#       ]
#     }
#   ],
#   "murs": {
#     "localisation": "",
#     "largeur": "",
#     "longueur": "",
#     "hauteur": "",
#     "description": ""
#   },
#   "escalier": {
#     "type": "",
#     "localisation": "",
#     "dimensions_m": "",
#     "direction": ""
#   },
#   "total_portes": "",
#   "total_fenetres": ""
# }
#                             """.strip()
                        
#                         if custom_prompt:
#                             base_prompt += f"\n\nInstructions supplémentaires : {custom_prompt}"
                        
#                         # Analyse avec Gemini avec timeout
#                         try:
#                             response = model.generate_content([base_prompt, image])
                            
#                             # Stocker les résultats
#                             st.session_state.analysis_result = response.text
#                             st.session_state.analysis_timestamp = datetime.now()
#                             st.session_state.parsed_data = parse_analysis_text(response.text)
                            
#                             st.success("✅ Analyse terminée avec succès!")
                            
#                         except Exception as e:
#                             st.error(f"⏱️ Timeout ou erreur: {str(e)}")
#                             st.info("💡 Essayez le mode rapide ou une image plus petite")
                        
#                     except Exception as e:
#                         st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
            
#             # Afficher les résultats si disponibles
#             if 'analysis_result' in st.session_state:
#                 st.markdown("---")
                
#                 # Métriques principales
#                 parsed_data = st.session_state.get('parsed_data', {"pieces_detaillees": [], "elements_techniques_globaux": {}})
                
#                 if parsed_data["pieces_detaillees"]:
#                     col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    
#                     pieces_valides = []
#                     total_portes = 0
#                     total_fenetres = 0
                    
#                     for piece in parsed_data["pieces_detaillees"]:
#                         if piece["surface_m2"] != "" and piece["surface_m2"] is not None:
#                             try:
#                                 pieces_valides.append(float(piece["surface_m2"]))
#                             except:
#                                 pass
                        
#                         if piece.get("nb_portes", "") != "":
#                             try:
#                                 total_portes += int(piece["nb_portes"])
#                             except:
#                                 pass
                        
#                         if piece.get("nb_fenetres", "") != "":
#                             try:
#                                 total_fenetres += int(piece["nb_fenetres"])
#                             except:
#                                 pass
                    
#                     with col_m1:
#                         st.metric("🏠 Pièces", len(parsed_data["pieces_detaillees"]))
                    
#                     with col_m2:
#                         if pieces_valides:
#                             st.metric("📐 Surface totale", f"{sum(pieces_valides):.1f} m²")
#                         else:
#                             st.metric("📐 Surface totale", "N/A")
                    
#                     with col_m3:
#                         # Utiliser le total global s'il existe, sinon calculer
#                         total_global = parsed_data["elements_techniques_globaux"].get("total_portes", "")
#                         if total_global:
#                             st.metric("🚪 Portes", total_global)
#                         else:
#                             st.metric("🚪 Portes", total_portes if total_portes > 0 else "N/A")
                    
#                     with col_m4:
#                         # Utiliser le total global s'il existe, sinon calculer
#                         total_global = parsed_data["elements_techniques_globaux"].get("total_fenetres", "")
#                         if total_global:
#                             st.metric("🪟 Fenêtres", total_global)
#                         else:
#                             st.metric("🪟 Fenêtres", total_fenetres if total_fenetres > 0 else "N/A")
    
#     # Affichage détaillé des résultats
#     if 'analysis_result' in st.session_state and 'parsed_data' in st.session_state:
#         st.markdown("---")
#         st.header("📊 Résultats Détaillés")
        
#         # Onglets pour les différentes vues
#         tab1, tab2, tab3 = st.tabs([
#             "📝 Analyse complète", 
#             "🔧 JSON", 
#             "💾 Téléchargements"
#         ])
        
#         with tab1:
#             st.subheader("📝 Analyse Complète")
#             st.text_area(
#                 "Rapport d'analyse détaillé",
#                 st.session_state.analysis_result,
#                 height=400,
#                 disabled=True
#             )
        
#         with tab2:
#             st.subheader("🔧 Structure JSON")
#             parsed_data = st.session_state.get('parsed_data', {})
#             st.json(parsed_data)
        
#         with tab3:
#             st.subheader("💾 Téléchargements")
#             st.write("Téléchargez les résultats dans différents formats :")
            
#             col_dl1, col_dl2 = st.columns(2)
            
#             with col_dl1:
#                 # Téléchargement JSON
#                 json_data = st.session_state.get('parsed_data', {})
#                 json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
                
#                 st.download_button(
#                     label="📄 JSON Structuré",
#                     data=json_str,
#                     file_name=f"analyse_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
#                     mime="application/json",
#                     help="Données structurées au format JSON"
#                 )
            
#             with col_dl2:
#                 # Téléchargement texte
#                 st.download_button(
#                     label="📝 Rapport TXT",
#                     data=st.session_state.analysis_result,
#                     file_name=f"analyse_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
#                     mime="text/plain",
#                     help="Analyse complète au format texte"
#                 )
            
#             # Informations sur l'analyse
#             st.markdown("---")
#             st.info(f"🕒 Analyse effectuée le: {st.session_state.analysis_timestamp.strftime('%d/%m/%Y à %H:%M:%S')}")
    
#     # Footer
#     st.markdown("---")
#     st.markdown(
#         """
#         <div style='text-align: center; color: gray; font-size: 0.9em;'>
#             🏗️ Analyseur de Plans d'Architecture - Powered by Google Gemini AI<br>
#             Développé avec Streamlit • Version 2.2
#         </div>
#         """, 
#         unsafe_allow_html=True
#     )

# if __name__ == "__main__":
#     main()