import google.generativeai as genai
import PIL.Image
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
import json
import streamlit as st

load_dotenv()


def check_api_configuration():
    """
    Vérifier que la clé API Gemini est correctement configurée
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Clé API Gemini non trouvée dans le fichier .env")
        print("📝 Veuillez ajouter GEMINI_API_KEY=votre_cle dans le fichier .env")
        return False
    else:
        print("✅ Clé API Gemini chargée depuis .env")
        return True


def extract_images_from_pdf(pdf_path):
    """
    Extraire les images d'un fichier PDF avec PyMuPDF
    Retourne une liste d'images PIL
    """
    print(f"📄 Extraction des images du PDF: {pdf_path}")
    try:
        # Ouvrir le PDF
        pdf_document = fitz.open(pdf_path)
        images = []
        
        # Parcourir toutes les pages
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Convertir la page en image avec haute résolution (300 DPI)
            # zoom = 300/72 = 4.166... (72 DPI par défaut)
            mat = fitz.Matrix(4.17, 4.17)
            pix = page.get_pixmap(matrix=mat)
            
            # Convertir en image PIL
            img_data = pix.tobytes("png")
            img = PIL.Image.open(io.BytesIO(img_data))
            images.append(img)
            
            print(f"✅ Page {page_num + 1}/{len(pdf_document)} extraite ({img.size})")
        
        pdf_document.close()
        print(f"✅ {len(images)} page(s) extraite(s) du PDF")
        return images
        
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction du PDF: {str(e)}")
        print("💡 Assurez-vous d'avoir installé PyMuPDF: pip install PyMuPDF")
        return None


def is_pdf_file(file_path):
    """
    Vérifier si le fichier est un PDF
    """
    return file_path.lower().endswith('.pdf')


def is_image_file(file_path):
    """
    Vérifier si le fichier est une image
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    return any(file_path.lower().endswith(ext) for ext in image_extensions)


def load_file(file_path):
    """
    Charger un fichier (image ou PDF) et retourner une liste d'images PIL
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas")
    
    if is_pdf_file(file_path):
        print("📄 Fichier PDF détecté")
        images = extract_images_from_pdf(file_path)
        if images is None:
            raise ValueError("Impossible d'extraire les images du PDF")
        return images
    
    elif is_image_file(file_path):
        print("🖼️ Fichier image détecté")
        return [PIL.Image.open(file_path)]
    
    else:
        raise ValueError(f"Format de fichier non supporté. Formats acceptés: PDF, JPG, PNG, BMP, TIFF")


def preprocess_architectural_plan(pil_image):
    """
    Prétraitement spécialisé pour les plans d'architecture techniques
    """
   
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

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

    # 6. Optimisation de la taille (important pour Gemini)
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
    """
    Amélioration spécifique pour les cotes et le texte des plans
    """
    edges = cv2.Canny(image, 30, 100, apertureSize=3)

    kernel_text = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(edges, kernel_text, iterations=1)

    kernel_close = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)

    enhanced = cv2.addWeighted(image, 0.85, closed, 0.15, 0)

    return enhanced


def optimize_image_size(image, max_size=1536):
    """
    Redimensionner l'image en gardant les proportions
    Optimisé pour Gemini (recommandé: 1024-2048px)
    """
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


def analyze_architectural_plan_with_preprocessing(file_path, api_key=None, save_preprocessed=True):
    """
    Analyse complète d'un plan d'architecture avec prétraitement optimisé
    Supporte les fichiers image et PDF
    
    Args:
        file_path: Chemin vers le fichier (image ou PDF)
        api_key: Clé API Gemini (optionnel, peut être chargée depuis .env)
        save_preprocessed: Sauvegarder les images prétraitées (défaut: True)
    
    Returns:
        Liste de résultats d'analyse (un par page/image)
    """

    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Clé API non trouvée. Veuillez la définir dans le fichier .env ou la passer en paramètre.")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    try:
        print("🔄 Chargement du fichier...")
        images = load_file(file_path)
        print(f"✅ {len(images)} image(s) chargée(s)")

        results = []

        for idx, original_image in enumerate(images, 1):
            print(f"\n{'='*50}")
            print(f"🔄 Traitement de l'image {idx}/{len(images)}")
            print(f"{'='*50}")
            print(f"📐 Taille originale: {original_image.size}")

            print("🔄 Prétraitement de l'image...")
            processed_image = preprocess_architectural_plan(original_image)
            print(f"✅ Image prétraitée: {processed_image.size}")

            if save_preprocessed:
                output_filename = f"plan_pretraite_{idx}.jpg"
                processed_image.save(output_filename, quality=95)
                print(f"💾 Image prétraitée sauvegardée: {output_filename}")

            # Prompt optimisé pour l'analyse
            prompt = """
            Analysez ce plan d'architecture prétraité et fournissez une synthèse rapide. 
Répondez uniquement au format JSON, sans texte supplémentaire. 
Donnez des descriptions uniques, concises et complètes afin que la génération prenne moins de 30 secondes.

Incluez :
- Pièces : une seule description par pièce qui combine le nom, la localisation, la surface approximative, les dimensions principales si visibles (longueur, largeur), les portes, les fenêtres et les équipements (prises, électroménager, mobilier, sanitaires… sans dimensions).
- Murs : une seule description par mur, qui combine la localisation, le type (porteur/cloison, intérieur/extérieur) et les dimensions principales (largeur, longueur, hauteur si visibles).
- Escalier : une seule description qui combine le type, la localisation et les dimensions approximatives.

Format JSON attendu :
{
  "pieces": [
    { "description": "Salon au centre, environ 25m², 5m x 5m, 2 portes, 1 fenêtre, équipé de canapé et table" }
  ],
  "murs": [
    { "description": "Mur nord, porteur intérieur, environ 4m de long, 0.3m de large, 2.5m de haut" }
  ],
  "escalier": {
    "description": "Escalier droit au sud, largeur environ 1m, montant vers l'étage"
  }
}

            """

            print("🔄 Analyse en cours avec Gemini...")
            response = model.generate_content([prompt, processed_image])

            print(f"✅ Analyse de l'image {idx} terminée!")
            results.append({
                'page': idx,
                'analysis': response.text
            })

        return results

    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Analyse de Plan", layout="wide")
    st.title("🏗️ Analyse de Plans Architecturaux avec La Bonne Réponse")

    # Upload fichier
    uploaded_file = st.file_uploader("📂 Importez un fichier (PDF ou image)", type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff"])
    
    if uploaded_file is not None:
        # Sauvegarde temporaire du fichier
        temp_path = uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ Fichier importé : {uploaded_file.name}")

        if st.button("🚀 Lancer l'analyse"):
            with st.spinner("Analyse en cours..."):
                results = analyze_architectural_plan_with_preprocessing(temp_path)

            if results:
                st.subheader("📊 Résultats de l'analyse")
                for result in results:
                    st.markdown(f"### 📄 Page/Image ")

                    try:
                        parsed = json.loads(result['analysis'])
                        st.json(parsed)

                        # Ajout d'une zone avec bouton de copie
                        st.code(json.dumps(parsed, indent=2, ensure_ascii=False), language="json")

                    except:
                        # st.text(result['analysis'])
                        st.code(result['analysis'], language="json")  # permet aussi de copier

            else:
                st.error("❌ Échec de l'analyse")
                
if __name__ == "__main__":
    print("🔧 Vérification de la configuration...")
    if not check_api_configuration():
        exit(1)
    
    # Vous pouvez maintenant utiliser un fichier PDF ou une image
    FILE_PATH = "Plan-2D.pdf"  # ou "Plan-2D.jpg"

    results = analyze_architectural_plan_with_preprocessing(FILE_PATH)

    if results:
        print("\n" + "="*60)
        print("RÉSULTATS DE L'ANALYSE:")
        print("="*60)
        
        for result in results:
            print(f"\n{'─'*60}")
            print(f"📄 PAGE/IMAGE :")
            print(f"{'─'*60}")
            # print(result['analysis'])
    else:
        print("❌ Échec de l'analyse")
    main()