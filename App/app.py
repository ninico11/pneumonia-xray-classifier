import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
from tensorflow import keras

# ---------- Paths & import ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "Model"

sys.path.append(str(MODEL_DIR))
from predict import PneumoniaPredictor  # clasa ta existentă


@st.cache_resource
def load_predictor():
    model_path_best = MODEL_DIR / "best_pneumonia_model.keras"
    model_path_final = MODEL_DIR / "pneumonia_detector_final.keras"

    if model_path_best.exists():
        return PneumoniaPredictor(model_path=str(model_path_best))
    elif model_path_final.exists():
        return PneumoniaPredictor(model_path=str(model_path_final))
    else:
        raise FileNotFoundError(
            "Nu am găsit nici 'best_pneumonia_model.keras' și nici "
            "'pneumonia_detector_final.keras' în folderul Model."
        )


predictor = load_predictor()
IMG_SIZE = predictor.img_size  # de ex. (224, 224)

st.set_page_config(page_title="Sistem de detectare a pneumoniei", layout="wide")

st.markdown(
    """
# 💀 Sistem de detectare a pneumoniei

Încarcă o imagine cu raze X a toracelui pentru o estimare automată a riscului de pneumonie.  
Acesta este doar un instrument de suport – diagnosticul final trebuie stabilit de un medic.
""",
)

st.markdown("---")

left_col, right_col = st.columns([2, 1])  

with left_col:
    st.subheader("Încarcă imagine (JPG sau PNG)")

    uploaded_file = st.file_uploader(
        "Selectează sau trage imaginea aici",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Imagine încărcată", use_container_width=True)

with right_col:
    st.subheader("Rezultat diagnostic (model)")
    st.write("Aici va apărea rezultatul după analiza imaginii.")

    result_box = st.container()

    analyze_clicked = st.button("🔍 Analizează imaginea", use_container_width=True)

    if analyze_clicked:
        if image is None:
            with result_box:
                st.warning("Te rog să încarci mai întâi o imagine cu raze X.")
        else:
            # Preprocesare
            img_resized = image.resize(IMG_SIZE)
            img_array = keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Predicție
            proba_pneumonia = predictor.model.predict(img_array, verbose=0)[0][0]
            is_pneumonia = proba_pneumonia > 0.5
            confidence = proba_pneumonia if is_pneumonia else 1 - proba_pneumonia
            predicted_class = "PNEUMONIE" if is_pneumonia else "NORMAL"

            # Afișare rezultate
            with result_box:
                st.markdown("### Rezultat diagnostic (model)")
                st.write(f"**Predicție:** {predicted_class}")
                st.write(f"**Încredere model:** {confidence:.2%}")
                st.write(
                    f"**Probabilitate pneumonie (ieșire model):** {proba_pneumonia:.4f}"
                )

                if predicted_class == "PNEUMONIE":
                    if confidence >= 0.9:
                        st.error(
                            "🔴 Probabilitate **RIDICATĂ** de pneumonie.\n\n"
                            "👉 Recomandare: consultație medicală **urgentă**."
                        )
                    elif confidence >= 0.7:
                        st.warning(
                            "🟠 Probabilitate **MODERATĂ** de pneumonie.\n\n"
                            "👉 Recomandare: consultație medicală în cel mai scurt timp."
                        )
                    else:
                        st.warning(
                            "🟡 Probabilitate **SCĂZUTĂ** de pneumonie.\n\n"
                            "👉 Recomandare: evaluare medicală pentru confirmare."
                        )
                else:
                    if confidence >= 0.9:
                        st.success(
                            "🟢 Probabilitate **FOARTE SCĂZUTĂ** de pneumonie.\n\n"
                            "👉 Plămânii par normali conform modelului."
                        )
                    elif confidence >= 0.7:
                        st.success(
                            "🟢 Probabilitate **SCĂZUTĂ** de pneumonie.\n\n"
                            "👉 Rezultat probabil negativ."
                        )
                    else:
                        st.warning(
                            "🟡 Rezultat **INCERT**.\n\n"
                            "👉 Recomandare: evaluare medicală pentru clarificare."
                        )

                st.caption(
                    "⚠️ Acest sistem este doar un instrument de asistență. "
                    "Nu înlocuiește consultul medical de specialitate."
                )
