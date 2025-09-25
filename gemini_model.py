import google.generativeai as genai
import PIL.Image
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Configuration avec la clé API depuis le fichier .env
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Charger le modèle
model = genai.GenerativeModel('gemini-2.5-flash')

# Charger votre plan d'architecture
image = PIL.Image.open("Plan-2D.jpg")

# Prompt spécialisé architecture
prompt = """
Analysez ce plan d'architecture et identifiez :
1. Toutes les pièces avec leurs dimensions approximatives
2. Mobilier et équipements (cuisine, salle de bain, électroménager)
3. Éléments techniques (portes, fenêtres, prises électriques)
4. Cotes et annotations visibles
5. Format JSON structuré pour traitement automatique
6. Liste des pièces avec leurs dimensions
7. liste le nombre des portes et des fenetres et des mures
"""

# Analyse
response = model.generate_content([prompt, image])
print(response.text)