import streamlit as st
import google.generativeai as genai
import PIL.Image
import json
import pandas as pd
from io import BytesIO
import re
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

# Fonction pour extraire les données structurées de l'analyse - ADAPTÉE AU NOUVEAU FORMAT
def parse_analysis_text(text):
    """Parse le texte d'analyse pour extraire les données selon le nouveau format JSON"""
    data = {
        "pieces_detaillees": [],
        "escaliers": [],
        "murs": [],
        "elements_techniques_globaux": {},
        "texte_complet": text
    }
    
    # Extraire le JSON s'il existe
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    json_data = None
    if json_match:
        try:
            json_data = json.loads(json_match.group(1))
        except:
            pass
    
    # Essayer de parser le JSON directement si pas de balises ```json
    if not json_data:
        try:
            json_data = json.loads(text.strip())
        except:
            pass
    
    # Parser selon le nouveau format JSON
    if json_data:
        # Extraire les pièces avec le nouveau format
        if "pieces" in json_data:
            for piece in json_data["pieces"]:
                # Extraire les dimensions
                dimensions = piece.get("dimensions_m", {})
                largeur = dimensions.get("largeur", "") if dimensions else ""
                longueur = dimensions.get("longueur", "") if dimensions else ""
                hauteur = dimensions.get("hauteur", "") if dimensions else ""
                
                # Compter et extraire les portes
                portes = piece.get("portes", [])
                nb_portes = len(portes) if portes else 0
                dim_portes = " | ".join([p.get("dimension_m", "") for p in portes if p.get("dimension_m", "")])
                
                # Compter et extraire les fenêtres
                fenetres = piece.get("fenetres", [])
                nb_fenetres = len(fenetres) if fenetres else 0
                dim_fenetres = " | ".join([f.get("dimension_m", "") for f in fenetres if f.get("dimension_m", "")])
                
                # Extraire les équipements électriques
                equipements_electriques = piece.get("equipements_electriques", [])
                nb_prises_electriques = len(equipements_electriques) if equipements_electriques else 0
                dim_prises_electriques = " | ".join([e.get("dimension_m", "") for e in equipements_electriques if e.get("dimension_m", "")])
                equipements_electriques_text = " | ".join([f"{e.get('nom', '')} ({e.get('dimension_m', '')})" for e in equipements_electriques])
                
                # Extraire les autres équipements
                autres_equipements = piece.get("autres_equipements", [])
                autres_equipements_text = " | ".join([f"{e.get('nom', '')} ({e.get('dimension_m', '')})" for e in autres_equipements])
                
                # Combiner tous les équipements
                equipements_total = []
                if equipements_electriques_text:
                    equipements_total.append(f"Électriques: {equipements_electriques_text}")
                if autres_equipements_text:
                    equipements_total.append(f"Autres: {autres_equipements_text}")
                
                piece_info = {
                    "piece": piece.get("nom", ""),
                    "localisation": piece.get("localisation", ""),
                    "surface_m2": piece.get("surface_m2", ""),
                    "largeur_m": largeur,
                    "longueur_m": longueur,
                    "hauteur_m": hauteur,
                    "equipements": " | ".join(equipements_total),
                    "nb_portes": nb_portes,
                    "nb_fenetres": nb_fenetres,
                    "nb_prises_electriques": nb_prises_electriques,
                    "dim_portes": dim_portes,
                    "dim_fenetres": dim_fenetres,
                    "dim_prises_electriques": dim_prises_electriques
                }
                
                data["pieces_detaillees"].append(piece_info)
        
        # Extraire les informations sur les murs
        if "murs" in json_data:
            murs_info = json_data["murs"]
            data["murs"].append({
                "localisation": murs_info.get("localisation", ""),
                "largeur": murs_info.get("largeur", ""),
                "longueur": murs_info.get("longueur", ""),
                "hauteur": murs_info.get("hauteur", ""),
                "description": murs_info.get("description", "")
            })
        
        # Extraire les informations sur l'escalier
        if "escalier" in json_data:
            escalier_info = json_data["escalier"]
            data["escaliers"].append({
                "type": escalier_info.get("type", ""),
                "localisation": escalier_info.get("localisation", ""),
                "dimensions": escalier_info.get("dimensions_m", ""),
                "direction": escalier_info.get("direction", "")
            })
        
        # Extraire les totaux
        data["elements_techniques_globaux"]["total_portes"] = json_data.get("total_portes", "")
        data["elements_techniques_globaux"]["total_fenetres"] = json_data.get("total_fenetres", "")
    
    else:
        # Parsing basique si pas de JSON (fallback)
        piece_pattern = r'(\w+(?:\s+\w+)*?):\s*(\d+[,\.]\d+)m²'
        pieces_matches = re.findall(piece_pattern, text, re.IGNORECASE)
        
        for nom, surface in pieces_matches:
            nom_clean = nom.strip().upper()
            surface_float = float(surface.replace(',', '.'))
            data["pieces_detaillees"].append({
                "piece": nom_clean,
                "localisation": "",
                "surface_m2": surface_float,
                "largeur_m": "",
                "longueur_m": "",
                "hauteur_m": "",
                "equipements": "",
                "nb_portes": "",
                "nb_fenetres": "",
                "nb_prises_electriques": "",
                "dim_portes": "",
                "dim_fenetres": "",
                "dim_prises_electriques": ""
            })
    
    return data

# Fonction pour créer le fichier Excel - ADAPTÉE AU NOUVEAU FORMAT
def create_enhanced_excel(data):
    """Crée un fichier Excel adapté au nouveau format JSON"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Formats
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#4CAF50',
            'font_color': 'white',
            'border': 1
        })
        
        cell_format = workbook.add_format({
            'text_wrap': True,
            'valign': 'top',
            'border': 1
        })
        
        number_format = workbook.add_format({
            'num_format': '0.00',
            'valign': 'top',
            'border': 1
        })
        
        # Créer la feuille principale
        worksheet = workbook.add_worksheet('Analyse_Plan')
        
        # En-têtes des colonnes adaptés au nouveau format
        headers = [
            'Nom Pièce',
            'Localisation',
            'Surface (m²)',
            'Largeur (m)',
            'Longueur (m)',
            'Hauteur (m)',
            'Équipements',
            'Nb Portes',
            'Dimensions Portes',
            'Nb Fenêtres',
            'Dimensions Fenêtres',
            'Nb Équip. Électriques',
            'Dimensions Équip. Électriques'
        ]
        
        # Écrire les en-têtes
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, header_format)
        
        # Définir les largeurs des colonnes
        column_widths = [20, 20, 12, 12, 12, 12, 40, 10, 25, 10, 25, 12, 25]
        for col, width in enumerate(column_widths):
            worksheet.set_column(col, col, width)
        
        # Remplir les données des pièces
        row = 1
        if data["pieces_detaillees"]:
            for piece in data["pieces_detaillees"]:
                # Nom de la pièce
                worksheet.write(row, 0, piece.get("piece", ""), cell_format)
                
                # Localisation
                worksheet.write(row, 1, piece.get("localisation", ""), cell_format)
                
                # Surface
                surface = piece.get("surface_m2", "")
                if surface != "" and surface is not None:
                    try:
                        surface_num = float(surface)
                        worksheet.write(row, 2, surface_num, number_format)
                    except:
                        worksheet.write(row, 2, surface, cell_format)
                else:
                    worksheet.write(row, 2, "", cell_format)
                
                # Largeur
                largeur = piece.get("largeur_m", "")
                if largeur != "" and largeur is not None:
                    try:
                        largeur_num = float(largeur)
                        worksheet.write(row, 3, largeur_num, number_format)
                    except:
                        worksheet.write(row, 3, largeur, cell_format)
                else:
                    worksheet.write(row, 3, "", cell_format)
                
                # Longueur
                longueur = piece.get("longueur_m", "")
                if longueur != "" and longueur is not None:
                    try:
                        longueur_num = float(longueur)
                        worksheet.write(row, 4, longueur_num, number_format)
                    except:
                        worksheet.write(row, 4, longueur, cell_format)
                else:
                    worksheet.write(row, 4, "", cell_format)
                
                # Hauteur
                hauteur = piece.get("hauteur_m", "")
                if hauteur != "" and hauteur is not None:
                    try:
                        hauteur_num = float(hauteur)
                        worksheet.write(row, 5, hauteur_num, number_format)
                    except:
                        worksheet.write(row, 5, hauteur, cell_format)
                else:
                    worksheet.write(row, 5, "", cell_format)
                
                # Équipements
                worksheet.write(row, 6, piece.get("equipements", ""), cell_format)
                
                # Nombre de portes
                nb_portes = piece.get("nb_portes", "")
                worksheet.write(row, 7, nb_portes, cell_format)
                
                # Dimensions portes
                worksheet.write(row, 8, piece.get("dim_portes", ""), cell_format)
                
                # Nombre de fenêtres
                nb_fenetres = piece.get("nb_fenetres", "")
                worksheet.write(row, 9, nb_fenetres, cell_format)
                
                # Dimensions fenêtres
                worksheet.write(row, 10, piece.get("dim_fenetres", ""), cell_format)
                
                # Nombre d'équipements électriques
                nb_prises = piece.get("nb_prises_electriques", "")
                worksheet.write(row, 11, nb_prises, cell_format)
                
                # Dimensions équipements électriques
                worksheet.write(row, 12, piece.get("dim_prises_electriques", ""), cell_format)
                
                row += 1
        
        # Section escaliers (si présents)
        if data["escaliers"]:
            row += 2
            worksheet.merge_range(row, 0, row, 12, "ESCALIERS", header_format)
            row += 1
            
            escalier_headers = ['Type', 'Localisation', 'Dimensions', 'Direction'] + [''] * 9
            for col, header in enumerate(escalier_headers):
                worksheet.write(row, col, header, header_format)
            row += 1
            
            for escalier in data["escaliers"]:
                worksheet.write(row, 0, escalier.get("type", ""), cell_format)
                worksheet.write(row, 1, escalier.get("localisation", ""), cell_format)
                worksheet.write(row, 2, escalier.get("dimensions", ""), cell_format)
                worksheet.write(row, 3, escalier.get("direction", ""), cell_format)
                for col in range(4, 13):
                    worksheet.write(row, col, "", cell_format)
                row += 1
        
        # Section murs (si présents)
        if data["murs"]:
            row += 2
            worksheet.merge_range(row, 0, row, 12, "MURS", header_format)
            row += 1
            
            mur_headers = ['Localisation', 'Largeur', 'Longueur', 'Hauteur', 'Description'] + [''] * 8
            for col, header in enumerate(mur_headers):
                worksheet.write(row, col, header, header_format)
            row += 1
            
            for mur in data["murs"]:
                worksheet.write(row, 0, mur.get("localisation", ""), cell_format)
                worksheet.write(row, 1, mur.get("largeur", ""), cell_format)
                worksheet.write(row, 2, mur.get("longueur", ""), cell_format)
                worksheet.write(row, 3, mur.get("hauteur", ""), cell_format)
                worksheet.write(row, 4, mur.get("description", ""), cell_format)
                for col in range(5, 13):
                    worksheet.write(row, col, "", cell_format)
                row += 1
        
        # Section résumé global
        if data["elements_techniques_globaux"]:
            row += 2
            worksheet.merge_range(row, 0, row, 12, "RÉSUMÉ GLOBAL", header_format)
            row += 1
            
            worksheet.write(row, 0, "Total Portes", header_format)
            worksheet.write(row, 1, data["elements_techniques_globaux"].get("total_portes", ""), cell_format)
            row += 1
            
            worksheet.write(row, 0, "Total Fenêtres", header_format)
            worksheet.write(row, 1, data["elements_techniques_globaux"].get("total_fenetres", ""), cell_format)
        
        # Feuille avec l'analyse complète
        worksheet_text = workbook.add_worksheet('Analyse_Complète')
        worksheet_text.write(0, 0, "Analyse Complète du Plan", header_format)
        worksheet_text.set_column(0, 0, 100)
        worksheet_text.write(1, 0, data["texte_complet"], cell_format)
        worksheet_text.set_row(1, 300)
    
    output.seek(0)
    return output

# Interface principale
def main():
    st.title("🏗️ Analyseur de Plans d'Architecture")
    st.markdown("*Analyse automatique avec IA - Détection d'objets et calcul de surfaces*")
    st.markdown("---")
    
    # Sidebar pour les paramètres
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Prompt personnalisé
        st.subheader("📝 Personnalisation du prompt")
        custom_prompt = st.text_area(
            "Prompt personnalisé (optionnel)",
            placeholder="Ajoutez des instructions spécifiques...",
            height=100
        )
        
        # Options d'analyse
        st.subheader("🔍 Options d'analyse")
        analyze_surfaces = st.checkbox("Calculer les surfaces", value=True)
        analyze_equipment = st.checkbox("Détecter les équipements", value=True)
        analyze_technical = st.checkbox("Compter les éléments techniques", value=True)
        json_output = st.checkbox("Sortie JSON structurée", value=True)
        
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
            st.image(image, caption="Plan d'architecture uploadé", width=None)
            
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
                        
                        # Construire le prompt
                        base_prompt = """
Analysez ce plan d'architecture et fournissez les informations suivantes :

Listez toutes les pièces avec :
- leur nom
- localisation
- surface (en m²)
- dimensions approximatives (longueur, largeur, hauteur si disponible)
- portes avec leurs dimensions
- fenêtres avec leurs dimensions
- équipements électriques avec dimensions si visibles (prises, VRE, etc.)
- autres équipements avec leurs dimensions (mobilier, électroménager, sanitaires…)

Décrivez les murs :
- localisation
- largeur
- longueur
- hauteur
- description (type intérieur/extérieur, porteur ou cloison, etc.)

Décrivez l'escalier :
- type
- localisation
- dimensions approximatives
- direction

Comptez le nombre de portes et fenêtres total

Répondez au format JSON suivant :
{
  "pieces": [
    {
      "nom": "",
      "localisation": "",
      "surface_m2": "",
      "dimensions_m": {
        "longueur": "",
        "largeur": "",
        "hauteur": ""
      },
      "portes": [
        { "dimension_m": "" }
      ],
      "fenetres": [
        { "dimension_m": "" }
      ],
      "equipements_electriques": [
        { "nom": "", "dimension_m": "" }
      ],
      "autres_equipements": [
        { "nom": "", "dimension_m": "" }
      ]
    }
  ],
  "murs": {
    "localisation": "",
    "largeur": "",
    "longueur": "",
    "hauteur": "",
    "description": ""
  },
  "escalier": {
    "type": "",
    "localisation": "",
    "dimensions_m": "",
    "direction": ""
  },
  "total_portes": "",
  "total_fenetres": ""
}
                        """.strip()
                        
                        if custom_prompt:
                            base_prompt += f"\n\nInstructions supplémentaires : {custom_prompt}"
                        
                        # Analyse avec Gemini
                        response = model.generate_content([base_prompt, image])
                        
                        # Stocker les résultats
                        st.session_state.analysis_result = response.text
                        st.session_state.analysis_timestamp = datetime.now()
                        st.session_state.parsed_data = parse_analysis_text(response.text)
                        
                        st.success("✅ Analyse terminée avec succès!")
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
            
            # Afficher les résultats si disponibles
            if 'analysis_result' in st.session_state:
                st.markdown("---")
                
                # Métriques principales
                parsed_data = st.session_state.get('parsed_data', {"pieces_detaillees": [], "elements_techniques_globaux": {}})
                
                if parsed_data["pieces_detaillees"]:
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    
                    pieces_valides = []
                    total_portes = 0
                    total_fenetres = 0
                    
                    for piece in parsed_data["pieces_detaillees"]:
                        if piece["surface_m2"] != "" and piece["surface_m2"] is not None:
                            try:
                                pieces_valides.append(float(piece["surface_m2"]))
                            except:
                                pass
                        
                        if piece.get("nb_portes", "") != "":
                            try:
                                total_portes += int(piece["nb_portes"])
                            except:
                                pass
                        
                        if piece.get("nb_fenetres", "") != "":
                            try:
                                total_fenetres += int(piece["nb_fenetres"])
                            except:
                                pass
                    
                    with col_m1:
                        st.metric("🏠 Pièces", len(parsed_data["pieces_detaillees"]))
                    
                    with col_m2:
                        if pieces_valides:
                            st.metric("📐 Surface totale", f"{sum(pieces_valides):.1f} m²")
                        else:
                            st.metric("📐 Surface totale", "N/A")
                    
                    with col_m3:
                        # Utiliser le total global s'il existe, sinon calculer
                        total_global = parsed_data["elements_techniques_globaux"].get("total_portes", "")
                        if total_global:
                            st.metric("🚪 Portes", total_global)
                        else:
                            st.metric("🚪 Portes", total_portes if total_portes > 0 else "N/A")
                    
                    with col_m4:
                        # Utiliser le total global s'il existe, sinon calculer
                        total_global = parsed_data["elements_techniques_globaux"].get("total_fenetres", "")
                        if total_global:
                            st.metric("🪟 Fenêtres", total_global)
                        else:
                            st.metric("🪟 Fenêtres", total_fenetres if total_fenetres > 0 else "N/A")
    
    # Affichage détaillé des résultats
    if 'analysis_result' in st.session_state and 'parsed_data' in st.session_state:
        st.markdown("---")
        st.header("📊 Résultats Détaillés")
        
        # Onglets pour les différentes vues
        tab1, tab2, tab3 = st.tabs([
            "📝 Analyse complète", 
            "🔧 JSON", 
            "💾 Téléchargements"
        ])
        
        with tab1:
            st.subheader("📝 Analyse Complète")
            st.text_area(
                "Rapport d'analyse détaillé",
                st.session_state.analysis_result,
                height=400,
                disabled=True
            )
        
        with tab2:
            st.subheader("🔧 Structure JSON")
            parsed_data = st.session_state.get('parsed_data', {})
            st.json(parsed_data)
        
        with tab3:
            st.subheader("💾 Téléchargements")
            st.write("Téléchargez les résultats dans différents formats :")
            
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            
            with col_dl1:
                # Téléchargement Excel avec structure corrigée
                parsed_data_for_excel = st.session_state.get('parsed_data', {})
                excel_data = create_enhanced_excel(parsed_data_for_excel)
                
                st.download_button(
                    label="📊 Excel Structuré",
                    data=excel_data,
                    file_name=f"analyse_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="Fichier Excel avec colonnes séparées pour chaque donnée"
                )
            
            with col_dl2:
                # Téléchargement JSON
                json_data = st.session_state.get('parsed_data', {})
                json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
                
                st.download_button(
                    label="📄 JSON Structuré",
                    data=json_str,
                    file_name=f"analyse_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    help="Données structurées au format JSON"
                )
            
            with col_dl3:
                # Téléchargement texte
                st.download_button(
                    label="📝 Rapport TXT",
                    data=st.session_state.analysis_result,
                    file_name=f"analyse_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    help="Analyse complète au format texte"
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
            Développé avec Streamlit • Version 2.2
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()