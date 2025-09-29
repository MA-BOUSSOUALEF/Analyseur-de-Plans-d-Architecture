import streamlit as st
import google.generativeai as genai
import openai
import PIL.Image
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import base64
import json
import time
from datetime import datetime
import io
from difflib import SequenceMatcher
import plotly.graph_objects as go
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

st.set_page_config(
    page_title="Comparaison Gemini vs GPT Vision - Plans Architecturaux",
    page_icon="🏗️",
    layout="wide"
)

# ==================== FONCTIONS DE PRÉTRAITEMENT ====================

def preprocess_architectural_plan(pil_image):
    """
    Prétraitement spécialisé pour les plans d'architecture techniques
    """
    # Convertir PIL vers OpenCV
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # 1. Conversion en niveaux de gris
    if len(cv_image.shape) == 3:
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv_image

    # 2. Amélioration du contraste adaptatif (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)

    # 3. Réduction du bruit tout en préservant les détails fins
    denoised = cv2.bilateralFilter(contrast_enhanced, 9, 75, 75)

    # 4. Amélioration de la netteté pour les lignes et le texte
    sharpen_kernel = np.array([[-1,-1,-1,-1,-1],
                              [-1, 2, 2, 2,-1],
                              [-1, 2, 8, 2,-1],
                              [-1, 2, 2, 2,-1],
                              [-1,-1,-1,-1,-1]]) / 8.0

    sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)

    # 5. Amélioration spécifique pour les cotes et annotations
    text_enhanced = enhance_text_and_dimensions(sharpened)

    # 6. Optimisation de la taille
    optimized = optimize_image_size(text_enhanced, max_size=1536)

    # 7. Conversion finale vers PIL
    processed_pil = Image.fromarray(optimized)

    # 8. Ajustement final de luminosité et contraste
    enhancer = ImageEnhance.Contrast(processed_pil)
    final_image = enhancer.enhance(1.2)

    enhancer = ImageEnhance.Brightness(final_image)
    final_image = enhancer.enhance(1.1)

    return final_image

def enhance_text_and_dimensions(image):
    """Amélioration spécifique pour les cotes et le texte des plans"""
    edges = cv2.Canny(image, 30, 100, apertureSize=3)
    kernel_text = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(edges, kernel_text, iterations=1)
    kernel_close = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
    enhanced = cv2.addWeighted(image, 0.85, closed, 0.15, 0)
    return enhanced

def optimize_image_size(image, max_size=1536):
    """Redimensionner l'image en gardant les proportions"""
    height, width = image.shape[:2]

    if max(height, width) > max_size:
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))

        resized = cv2.resize(image, (new_width, new_height),
                           interpolation=cv2.INTER_AREA)
        return resized

    return image

# ==================== PROMPT ARCHITECTURAL ====================

ARCHITECTURAL_PROMPT = """
Analysez ce plan d'architecture prétraité et fournissez les informations suivantes :

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
      "tous_les_equipements": [
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
"""

# ==================== INTERFACE STREAMLIT ====================

st.title("🏗️ Comparateur Gemini vs GPT-4 Vision pour Plans Architecturaux")
st.markdown("**Avec prétraitement d'image optimisé pour l'analyse de plans**")

# Configuration dans la sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Charger les clés depuis .env si disponibles
    env_gemini_key = os.getenv("GEMINI_API_KEY")
    env_openai_key = os.getenv("OPENAI_API_KEY")
    
    # Initialiser les clés dans session_state si elles existent dans .env
    if env_gemini_key and 'gemini_key' not in st.session_state:
        st.session_state.gemini_key = env_gemini_key
    if env_openai_key and 'openai_key' not in st.session_state:
        st.session_state.openai_key = env_openai_key
    
    # Afficher le statut des clés chargées depuis .env
    if env_gemini_key:
        st.success("✅ Clé Gemini chargée depuis .env")
    if env_openai_key:
        st.success("✅ Clé OpenAI chargée depuis .env")
    
    # Permettre la saisie manuelle des clés (optionnel si pas dans .env)
    gemini_placeholder = "Déjà chargée depuis .env" if env_gemini_key else "Entrez votre clé API Gemini"
    openai_placeholder = "Déjà chargée depuis .env" if env_openai_key else "Entrez votre clé API OpenAI"
    
    gemini_key = st.text_input("Clé API Gemini", type="password", key="gemini_input", placeholder=gemini_placeholder)
    openai_key = st.text_input("Clé API OpenAI", type="password", key="openai_input", placeholder=openai_placeholder)
    
    if st.button("💾 Sauvegarder les clés manuelles"):
        if gemini_key and openai_key:
            st.session_state.gemini_key = gemini_key
            st.session_state.openai_key = openai_key
            st.success("Clés manuelles sauvegardées!")
        elif gemini_key:
            st.session_state.gemini_key = gemini_key
            st.success("Clé Gemini sauvegardée!")
        elif openai_key:
            st.session_state.openai_key = openai_key
            st.success("Clé OpenAI sauvegardée!")
    
    st.divider()
    st.subheader("🔧 Options de prétraitement")
    show_preprocessed = st.checkbox("Afficher l'image prétraitée", value=True)
    download_preprocessed = st.checkbox("Permettre le téléchargement de l'image prétraitée", value=True)

# Upload d'image
uploaded_file = st.file_uploader(
    "📁 Uploadez votre plan architectural",
    type=['jpg', 'jpeg', 'png'],
    help="Formats supportés: JPG, JPEG, PNG"
)

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Image originale")
        original_image = PIL.Image.open(uploaded_file)
        st.image(original_image, caption=f"Taille: {original_image.size}", use_column_width=True)
    
    with col2:
        st.subheader("🔧 Image prétraitée")
        with st.spinner("Prétraitement en cours..."):
            processed_image = preprocess_architectural_plan(original_image)
            
            if show_preprocessed:
                st.image(processed_image, caption=f"Taille: {processed_image.size}", use_column_width=True)
                st.success("✅ Prétraitement terminé!")
                
                # Afficher les étapes du prétraitement
                with st.expander("ℹ️ Détails du prétraitement"):
                    st.write("""
                    **Étapes appliquées:**
                    1. ✓ Conversion en niveaux de gris
                    2. ✓ Amélioration du contraste adaptatif (CLAHE)
                    3. ✓ Réduction du bruit (filtre bilatéral)
                    4. ✓ Amélioration de la netteté
                    5. ✓ Renforcement du texte et des cotes
                    6. ✓ Optimisation de la taille (max 1536px)
                    7. ✓ Ajustement luminosité/contraste
                    """)
            
            # Bouton de téléchargement de l'image prétraitée
            if download_preprocessed:
                buffered = io.BytesIO()
                processed_image.save(buffered, format="JPEG", quality=95)
                st.download_button(
                    label="💾 Télécharger l'image prétraitée",
                    data=buffered.getvalue(),
                    file_name="plan_pretraite.jpg",
                    mime="image/jpeg"
                )

# Fonctions d'analyse avec prétraitement
@st.cache_data(ttl=300, show_spinner=False)
def analyze_with_gemini(processed_image_bytes, prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Convertir bytes en PIL Image
        pil_image = PIL.Image.open(io.BytesIO(processed_image_bytes))
        
        start_time = time.time()
        response = model.generate_content([prompt, pil_image])
        end_time = time.time()
        
        return {
            "success": True,
            "response": response.text,
            "response_time": end_time - start_time,
            "model": "gemini-2.5-flash"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model": "gemini-2.5-flash"
        }

@st.cache_data(ttl=300, show_spinner=False)
def analyze_with_gpt(processed_image_bytes, prompt, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        base64_image = base64.b64encode(processed_image_bytes).decode('utf-8')
        
        start_time = time.time()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }],
            max_tokens=2000
        )
        end_time = time.time()
        
        return {
            "success": True,
            "response": response.choices[0].message.content,
            "response_time": end_time - start_time,
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "model": "gpt-4o"
        }

def calculate_similarity(text1, text2):
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def parse_json_response(text):
    """Tente de parser une réponse JSON"""
    try:
        # Nettoyer le texte
        cleaned = text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(cleaned), True
    except:
        return None, False

# Bouton de lancement de l'analyse
st.divider()
if st.button("🚀 Lancer la comparaison", type="primary", use_container_width=True):
    if not uploaded_file:
        st.error("❌ Veuillez d'abord uploader une image")
    elif 'gemini_key' not in st.session_state or 'openai_key' not in st.session_state:
        st.error("❌ Veuillez configurer vos clés API dans la barre latérale")
    else:
        # Convertir l'image prétraitée en bytes
        buffered = io.BytesIO()
        processed_image.save(buffered, format="JPEG", quality=95)
        processed_image_bytes = buffered.getvalue()
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Analyse avec Gemini
        status_text.text("🤖 Analyse avec Gemini 2.5 Flash...")
        progress_bar.progress(25)
        
        gemini_result = analyze_with_gemini(
            processed_image_bytes,
            ARCHITECTURAL_PROMPT,
            st.session_state.gemini_key
        )
        
        progress_bar.progress(50)
        
        # Analyse avec GPT-4
        status_text.text("🧠 Analyse avec GPT-4 Vision...")
        gpt_result = analyze_with_gpt(
            processed_image_bytes,
            ARCHITECTURAL_PROMPT,
            st.session_state.openai_key
        )
        
        progress_bar.progress(75)
        status_text.text("📊 Traitement des résultats...")
        
        # Compilation des résultats
        results = {
            "timestamp": datetime.now().isoformat(),
            "original_size": original_image.size,
            "processed_size": processed_image.size,
            "prompt": ARCHITECTURAL_PROMPT,
            "gemini": gemini_result,
            "gpt": gpt_result
        }
        
        progress_bar.progress(100)
        status_text.text("✅ Analyse terminée!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # ==================== AFFICHAGE DES RÉSULTATS ====================
        
        st.header("📊 Résultats de la comparaison")
        
        # Métriques globales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if gemini_result["success"]:
                st.metric("⏱️ Temps Gemini", f"{gemini_result['response_time']:.2f}s")
            else:
                st.metric("⏱️ Temps Gemini", "Erreur", delta_color="off")
        
        with col2:
            if gpt_result["success"]:
                st.metric("⏱️ Temps GPT-4", f"{gpt_result['response_time']:.2f}s")
            else:
                st.metric("⏱️ Temps GPT-4", "Erreur", delta_color="off")
        
        with col3:
            if gemini_result["success"] and gpt_result["success"]:
                similarity = calculate_similarity(
                    gemini_result["response"],
                    gpt_result["response"]
                )
                st.metric("🔍 Similarité", f"{similarity:.1%}")
            else:
                st.metric("🔍 Similarité", "N/A", delta_color="off")
        
        with col4:
            success_count = sum([gemini_result["success"], gpt_result["success"]])
            st.metric("✅ Analyses réussies", f"{success_count}/2")
        
        # Graphique de comparaison des temps
        if gemini_result["success"] and gpt_result["success"]:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Temps de réponse',
                x=['Gemini 2.5 Flash', 'GPT-4o'],
                y=[gemini_result['response_time'], gpt_result['response_time']],
                marker_color=['#4285F4', '#10A37F'],
                text=[f"{gemini_result['response_time']:.2f}s", f"{gpt_result['response_time']:.2f}s"],
                textposition='auto',
            ))
            fig.update_layout(
                title="⏱️ Comparaison des temps de réponse",
                yaxis_title="Temps (secondes)",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Affichage côte à côte des réponses
        st.divider()
        st.subheader("📋 Réponses détaillées")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🤖 Gemini 2.5 Flash")
            if gemini_result["success"]:
                st.info(f"✅ Succès | ⏱️ {gemini_result['response_time']:.2f}s | 📝 {len(gemini_result['response'])} caractères")
                
                # Tentative de parsing JSON
                gemini_json, gemini_valid = parse_json_response(gemini_result["response"])
                if gemini_valid:
                    st.success("✅ Format JSON valide")
                    with st.expander("📊 Données structurées"):
                        st.json(gemini_json)
                    
                    # Statistiques
                    if "pieces" in gemini_json:
                        st.metric("🏠 Pièces détectées", len(gemini_json["pieces"]))
                else:
                    st.warning("⚠️ Format JSON invalide ou absent")
                
                with st.expander("📄 Réponse complète", expanded=True):
                    st.text_area("", gemini_result["response"], height=400, key="gemini_response")
            else:
                st.error(f"❌ Erreur: {gemini_result['error']}")
        
        with col2:
            st.markdown("### 🧠 GPT-4o")
            if gpt_result["success"]:
                st.info(f"✅ Succès | ⏱️ {gpt_result['response_time']:.2f}s | 📝 {len(gpt_result['response'])} caractères")
                
                # Tokens utilisés
                if "usage" in gpt_result:
                    st.caption(f"🎫 Tokens: {gpt_result['usage']['total_tokens']} (prompt: {gpt_result['usage']['prompt_tokens']}, completion: {gpt_result['usage']['completion_tokens']})")
                
                # Tentative de parsing JSON
                gpt_json, gpt_valid = parse_json_response(gpt_result["response"])
                if gpt_valid:
                    st.success("✅ Format JSON valide")
                    with st.expander("📊 Données structurées"):
                        st.json(gpt_json)
                    
                    # Statistiques
                    if "pieces" in gpt_json:
                        st.metric("🏠 Pièces détectées", len(gpt_json["pieces"]))
                else:
                    st.warning("⚠️ Format JSON invalide ou absent")
                
                with st.expander("📄 Réponse complète", expanded=True):
                    st.text_area("", gpt_result["response"], height=400, key="gpt_response")
            else:
                st.error(f"❌ Erreur: {gpt_result['error']}")
        
        # Analyse comparative détaillée
        if gemini_result["success"] and gpt_result["success"]:
            st.divider()
            st.subheader("📈 Analyse comparative")
            
            gemini_json, gemini_valid = parse_json_response(gemini_result["response"])
            gpt_json, gpt_valid = parse_json_response(gpt_result["response"])
            
            if gemini_valid and gpt_valid:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    gemini_pieces = len(gemini_json.get("pieces", []))
                    gpt_pieces = len(gpt_json.get("pieces", []))
                    st.metric(
                        "🏠 Pièces détectées",
                        f"{gemini_pieces} vs {gpt_pieces}",
                        delta=gemini_pieces - gpt_pieces
                    )
                
                with col2:
                    gemini_portes = gemini_json.get("total_portes", "N/A")
                    gpt_portes = gpt_json.get("total_portes", "N/A")
                    st.metric("🚪 Total portes", f"G: {gemini_portes} | GPT: {gpt_portes}")
                
                with col3:
                    gemini_fenetres = gemini_json.get("total_fenetres", "N/A")
                    gpt_fenetres = gpt_json.get("total_fenetres", "N/A")
                    st.metric("🪟 Total fenêtres", f"G: {gemini_fenetres} | GPT: {gpt_fenetres}")
        
        # Export des résultats
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            results_json = json.dumps(results, indent=2, ensure_ascii=False)
            st.download_button(
                label="💾 Télécharger les résultats (JSON)",
                data=results_json,
                file_name=f"comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Créer un rapport texte
            report = f"""RAPPORT DE COMPARAISON - Plans Architecturaux
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

CONFIGURATION:
- Taille image originale: {original_image.size}
- Taille image prétraitée: {processed_image.size}

RÉSULTATS:
Gemini 2.5 Flash: {'✅ Succès' if gemini_result['success'] else '❌ Échec'}
  Temps: {gemini_result.get('response_time', 0):.2f}s
  Longueur: {len(gemini_result.get('response', ''))} caractères

GPT-4o: {'✅ Succès' if gpt_result['success'] else '❌ Échec'}
  Temps: {gpt_result.get('response_time', 0):.2f}s
  Longueur: {len(gpt_result.get('response', ''))} caractères

{'='*60}
RÉPONSE GEMINI:
{gemini_result.get('response', 'N/A')}

{'='*60}
RÉPONSE GPT-4:
{gpt_result.get('response', 'N/A')}
"""
            st.download_button(
                label="📄 Télécharger le rapport (TXT)",
                data=report,
                file_name=f"rapport_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <p>🏗️ Comparateur de Plans Architecturaux | Gemini 2.5 Flash vs GPT-4o</p>
    <p>Avec prétraitement d'image optimisé (CLAHE, débruitage, renforcement)</p>
</div>
""", unsafe_allow_html=True)