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
import streamlit.components.v1 as components
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
    
    client = openai.OpenAI(api_key=api_key)

    try:
        images = load_file(uploaded_file)
        results = []

        for idx, original_image in enumerate(images, 1):
            processed_image = preprocess_architectural_plan(original_image)

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

# --- Intégration Chatbase ---
def add_chatbase_widget(analysis_data=None, send_message=False):
    """
    Ajoute le widget Chatbase à l'application Streamlit
    
    Args:
        analysis_data: Données d'analyse à envoyer au chatbot
        send_message: Si True, envoie automatiquement les résultats au chatbot
    """
    # Préparer le message à envoyer
    message_to_send = ""
    if analysis_data and send_message and 'results' in analysis_data:
        message_to_send = f"Voici l'analyse du fichier '{analysis_data.get('filename', 'plan')}' :\n\n"
        for result in analysis_data['results']:
            message_to_send += f"Page {result['page']}:\n{result['analysis']}\n\n"
        
        # Encoder le message en base64
        message_encoded = base64.b64encode(message_to_send.encode('utf-8')).decode('utf-8')
    else:
        message_encoded = ""
    
    # Générer une clé unique pour forcer le rechargement
    unique_key = hash(message_to_send) if message_to_send else 0
    
    chatbase_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                height: 100%;
            }}
            .copy-notification {{
                position: fixed;
                top: 10px;
                right: 10px;
                background: #4CAF50;
                color: white;
                padding: 15px;
                border-radius: 5px;
                display: none;
                z-index: 10000;
            }}
        </style>
    </head>
    <body>
        <div id="copy-notification" class="copy-notification">✓ Message copié dans le presse-papiers !</div>
        <div id="chatbase-container-{unique_key}"></div>
        
        <script>
        window.embeddedChatbotConfig = {{
            chatbotId: "xuQdL4-FaXKMZBx9l1yN9",
            domain: "www.chatbase.co"
        }}
        </script>
        <script
            src="https://www.chatbase.co/embed.min.js"
            chatbotId="xuQdL4-FaXKMZBx9l1yN9"
            domain="www.chatbase.co"
            defer>
        </script>
        
        <script>
        const messageToSend = "{message_encoded}";
        
        // Fonction pour copier dans le presse-papiers
        window.copyAnalysisToClipboard = function() {{
            if (messageToSend) {{
                const decodedMessage = atob(messageToSend);
                navigator.clipboard.writeText(decodedMessage).then(() => {{
                    const notification = document.getElementById('copy-notification');
                    notification.style.display = 'block';
                    setTimeout(() => {{
                        notification.style.display = 'none';
                    }}, 3000);
                }}).catch(err => {{
                    console.error('Erreur de copie:', err);
                    alert('Erreur lors de la copie. Veuillez copier manuellement.');
                }});
            }}
        }}
        
        function waitForChatbase() {{
            return new Promise((resolve) => {{
                const checkChatbase = setInterval(() => {{
                    if (window.chatbase) {{
                        clearInterval(checkChatbase);
                        resolve();
                    }}
                }}, 100);
            }});
        }}
        
        async function sendMessageToChatbot() {{
            await waitForChatbase();
            
            // Attendre que le chatbot soit complètement chargé
            await new Promise(resolve => setTimeout(resolve, 3000));
            
            if (messageToSend && window.chatbase) {{
                try {{
                    const decodedMessage = atob(messageToSend);
                    console.log("Envoi du message au chatbot...");
                    
                    // Essayer de trouver le champ de saisie du chatbot et y insérer le texte
                    const inputField = document.querySelector('textarea[placeholder*="message"], textarea[placeholder*="Message"], input[type="text"]');
                    if (inputField) {{
                        inputField.value = decodedMessage;
                        inputField.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        console.log("Message inséré dans le champ de saisie");
                    }}
                    
                    // Différentes méthodes pour envoyer le message selon l'API Chatbase
                    if (typeof window.chatbase === 'function') {{
                        window.chatbase('sendMessage', decodedMessage);
                    }} else if (window.chatbase.sendMessage) {{
                        window.chatbase.sendMessage(decodedMessage);
                    }} else if (window.chatbase.send) {{
                        window.chatbase.send(decodedMessage);
                    }}
                    
                    console.log("Message envoyé avec succès!");
                }} catch (error) {{
                    console.error("Erreur lors de l'envoi:", error);
                }}
            }}
        }}
        
        // Lancer l'envoi du message
        if (messageToSend) {{
            sendMessageToChatbot();
        }}
        </script>
    </body>
    </html>
    """
    components.html(chatbase_html, height=600, scrolling=True)

# --- Streamlit ---
def main():
    st.set_page_config(page_title="Analyse de Plan", layout="wide")
    
    # Initialiser le state pour stocker les résultats
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'uploaded_filename' not in st.session_state:
        st.session_state.uploaded_filename = None
    
    # Créer deux colonnes
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("🏗️ Analyse de Plans Architecturaux avec La Bonne Réponse")

        if not check_api_configuration():
            return

        uploaded_file = st.file_uploader("📂 Importez un fichier (PDF ou image)", type=["pdf", "jpg", "jpeg", "png", "bmp", "tiff"])
        
        if uploaded_file is not None:
            st.success(f"✅ Fichier importé : {uploaded_file.name}")
            st.session_state.uploaded_filename = uploaded_file.name

            if st.button("🚀 Lancer l'analyse"):
                with st.spinner("Analyse en cours..."):
                    results = analyze_architectural_plan_with_gpt(uploaded_file)

                if results:
                    # Stocker les résultats dans le session state
                    st.session_state.analysis_results = results
                    
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
    
    with col2:
        st.markdown("### 💬 Assistant Chatbot")
        
        # Vérifier si une analyse a été effectuée
        if st.session_state.analysis_results:
            st.success(f"✅ Analyse disponible ({len(st.session_state.analysis_results)} page(s))")
            
            # Préparer le message JSON formaté
            json_message = f"Voici l'analyse du fichier '{st.session_state.uploaded_filename}' :\n\n"
            for result in st.session_state.analysis_results:
                json_message += f"📄 Page {result['page']}:\n```json\n{result['analysis']}\n```\n\n"
            
            # Boutons d'action
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                # Bouton pour copier dans le presse-papiers
                if st.button("📋 Copier JSON", use_container_width=True):
                    st.code(json_message, language="text")
                    st.info("👆 Copiez ce texte et collez-le dans le chatbot")
            
            with col_btn2:
                # Bouton pour envoyer automatiquement
                if st.button("📤 Envoyer au chat", use_container_width=True):
                    st.session_state.send_to_chat = True
                    st.rerun()
            
            # Zone de texte pour copier manuellement
            with st.expander("📝 Voir le message complet"):
                st.text_area(
                    "Message à envoyer au chatbot:",
                    value=json_message,
                    height=200,
                    key="json_textarea"
                )
                st.caption("💡 Copiez ce texte (Ctrl+A puis Ctrl+C) et collez-le dans le chatbot")
            
            # Préparer les données pour le chatbot
            chatbot_data = {
                'filename': st.session_state.uploaded_filename,
                'page_count': len(st.session_state.analysis_results),
                'results': st.session_state.analysis_results
            }
            
            # Vérifier si on doit envoyer le message automatiquement
            send_now = st.session_state.get('send_to_chat', False)
            if send_now:
                st.info("📤 Message envoyé au chatbot !")
                st.session_state.send_to_chat = False
            
            # Ajouter le widget Chatbase
            add_chatbase_widget(analysis_data=chatbot_data, send_message=send_now)
        else:
            st.info("💡 Uploadez et analysez un plan pour commencer")
            # Afficher le chatbot sans message
            add_chatbase_widget(analysis_data=None, send_message=False)

if __name__ == "__main__":
    main()