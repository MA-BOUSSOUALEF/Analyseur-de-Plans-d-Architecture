# === Ton code existant ci-dessus inchangé ===
# (ne modifie rien jusqu'à la fin de ton script)

import streamlit as st
import json
from Brique_Plan_pdf_and_image import *

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
    main()
