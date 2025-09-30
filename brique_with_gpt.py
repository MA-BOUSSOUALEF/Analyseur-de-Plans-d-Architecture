import openai
import PIL.Image
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
import json
import streamlit as st
import base64

load_dotenv()

def check_api_configuration():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ Clé API OpenAI non trouvée dans .env")
        return False
    return True

# --- Extraction et vérification fichiers ---
def extract_images_from_pdf(pdf_bytes):
    """
    Extraire les images d'un PDF à partir de bytes et retourner une liste d'images PIL
    """
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        mat = fitz.Matrix(4.17, 4.17)  # haute résolution 300 DPI
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        img = PIL.Image.open(io.BytesIO(img_data))
        images.append(img)
    pdf_document.close()
    return images

def is_pdf_file(filename):
    return filename.lower().endswith('.pdf')

def is_image_file(filename):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def load_file(uploaded_file):
    """
    Retourne une liste d'images PIL à partir d'un fichier uploadé (PDF ou image)
    """
    if is_pdf_file(uploaded_file.name):
        pdf_bytes = uploaded_file.read()
        return extract_images_from_pdf(pdf_bytes)
    elif is_image_file(uploaded_file.name):
        return [PIL.Image.open(uploaded_file)]
    else:
        raise ValueError("Format non supporté. Formats acceptés : PDF, JPG, PNG, BMP, TIFF")

# --- Prétraitement image ---
def enhance_text_and_dimensions(image):
    edges = cv2.Canny(image, 30, 100, apertureSize=3)
    kernel_text = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(edges, kernel_text, iterations=1)
    kernel_close = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
    enhanced = cv2.addWeighted(image, 0.85, closed, 0.15, 0)
    return enhanced

def optimize_image_size(image, max_size=1536):
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

def preprocess_architectural_plan(pil_image):
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) if len(cv_image.shape) == 3 else cv_image
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(contrast_enhanced, 9, 75, 75)
    sharpen_kernel = np.array([[-1,-1,-1,-1,-1],
                              [-1, 2, 2, 2,-1],
                              [-1, 2, 8, 2,-1],
                              [-1, 2, 2, 2,-1],
                              [-1,-1,-1,-1,-1]]) / 8.0
    sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)
    text_enhanced = enhance_text_and_dimensions(sharpened)
    optimized = optimize_image_size(text_enhanced, max_size=1536)
    processed_pil = Image.fromarray(optimized)
    enhancer = ImageEnhance.Contrast(processed_pil)
    final_image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Brightness(final_image)
    return enhancer.enhance(1.1)

# --- Analyse GPT ---
def analyze_architectural_plan_with_gpt(uploaded_file, api_key=None, save_preprocessed=True):
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Clé API OpenAI non trouvée. Définir OPENAI_API_KEY dans .env")
    
    # Créer le client OpenAI avec la nouvelle API
    client = openai.OpenAI(api_key=api_key)

    try:
        images = load_file(uploaded_file)
        results = []

        for idx, original_image in enumerate(images, 1):
            processed_image = preprocess_architectural_plan(original_image)

            # Encode the processed image to base64
            buffered = io.BytesIO()
            processed_image.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

            if save_preprocessed:
                output_filename = f"plan_pretraite_{idx}.jpg"
                processed_image.save(output_filename, quality=95)

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

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
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
                    }
                ],
                max_tokens=4000,
                temperature=0.1
            )

            analysis_text = response.choices[0].message.content
            results.append({'page': idx, 'analysis': analysis_text})

        return results

    except Exception as e:
        st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
        return None

# --- Streamlit ---
def main():
    st.set_page_config(page_title="Analyse de Plan", layout="wide")
    st.title("🏗️ Analyse de Plans Architecturaux avec La Bonne Réponse")

    if not check_api_configuration():
        return

    uploaded_file = st.file_uploader("📂 Importez un fichier (PDF ou image)", type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff"])
    
    if uploaded_file is not None:
        st.success(f"✅ Fichier importé : {uploaded_file.name}")

        if st.button("🚀 Lancer l'analyse"):
            with st.spinner("Analyse en cours..."):
                results = analyze_architectural_plan_with_gpt(uploaded_file)

            if results:
                st.subheader("📊 Résultats de l'analyse")
                for result in results:
                    st.markdown(f"### 📄 Page/Image {result['page']}")
                    try:
                        parsed = json.loads(result['analysis'])
                        st.json(parsed)
                        st.code(json.dumps(parsed, indent=2, ensure_ascii=False), language="json")
                    except json.JSONDecodeError as e:
                        st.warning(f"⚠️ Réponse non-JSON détectée pour la page {result['page']}")
                        st.code(result['analysis'], language="text")
            else:
                st.error("❌ Échec de l'analyse")

if __name__ == "__main__":
    main()
