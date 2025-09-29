import google.generativeai as genai
import PIL.Image
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import os
from dotenv import load_dotenv

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
    # Spécialement efficace pour les plans avec du texte et des lignes fines
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)

    # 3. Réduction du bruit tout en préservant les détails fins
    # Utilisation d'un filtre bilatéral qui préserve les contours
    denoised = cv2.bilateralFilter(contrast_enhanced, 9, 75, 75)

    # 4. Amélioration de la netteté pour les lignes et le texte
    # Kernel spécialement conçu pour renforcer les lignes fines
    sharpen_kernel = np.array([[-1,-1,-1,-1,-1],
                              [-1, 2, 2, 2,-1],
                              [-1, 2, 8, 2,-1],
                              [-1, 2, 2, 2,-1],
                              [-1,-1,-1,-1,-1]]) / 8.0

    sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)

    # 5. Amélioration spécifique pour les cotes et annotations
    # Création d'un masque pour renforcer le texte
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
    # Détection des contours pour identifier les lignes et le texte
    edges = cv2.Canny(image, 30, 100, apertureSize=3)

    # Dilatation légère pour connecter les caractères fragmentés
    kernel_text = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(edges, kernel_text, iterations=1)

    # Fermeture pour remplir les trous dans le texte
    kernel_close = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)

    # Combiner avec l'image originale pour renforcer le texte
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

        # Utilisation d'interpolation INTER_AREA pour la réduction (meilleure qualité)
        resized = cv2.resize(image, (new_width, new_height),
                           interpolation=cv2.INTER_AREA)
        return resized

    return image

def analyze_architectural_plan_with_preprocessing(image_path, api_key=None):
    """
    Analyse complète d'un plan d'architecture avec prétraitement optimisé
    Si api_key n'est pas fournie, elle sera chargée depuis le fichier .env
    """

    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Clé API non trouvée. Veuillez la définir dans le fichier .env ou la passer en paramètre.")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    try:
        print("🔄 Chargement de l'image...")
        original_image = PIL.Image.open(image_path)
        print(f"✅ Image chargée: {original_image.size}")

        print("🔄 Prétraitement de l'image...")
        processed_image = preprocess_architectural_plan(original_image)
        print(f"✅ Image prétraitée: {processed_image.size}")

        # processed_image.save("plan_pretraite.jpg", quality=95)

        # Prompt optimisé pour l'analyse
        prompt = """
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
          }
        }
        """

        # Analyse avec Gemini
        print("🔄 Analyse en cours avec Gemini...")
        processed_image.save("plan_pretraite.jpg", quality=95)
        print("✅ Image prétraitée sauvegardée : plan_pretraite.jpg")
        response = model.generate_content([prompt, processed_image])

        print("✅ Analyse terminée!")
        return response.text

    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {str(e)}")
        return None

if __name__ == "__main__":
    print("🔧 Vérification de la configuration...")
    if not check_api_configuration():
        exit(1)
    
    IMAGE_PATH = "Plan-2D.jpg"  

    result = analyze_architectural_plan_with_preprocessing(IMAGE_PATH)

    if result:
        print("\n" + "="*50)
        print("RÉSULTAT DE L'ANALYSE:")
        print("="*50)
        print(result)
    else:
        print("❌ Échec de l'analyse")

