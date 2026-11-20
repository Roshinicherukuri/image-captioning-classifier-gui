import csv
import io
import time
from datetime import datetime

import streamlit as st
from PIL import Image

from src.captioner import MODEL_PATH, TOKENIZER_PATH, ImageCaptioner

st.set_page_config(page_title="Image Captioning Classifier", page_icon="🖼️", layout="wide")

st.markdown(
    """
    <style>
      .caption-box {padding: .75rem 1rem; border-radius: .6rem; margin-top: .5rem;
                    background: rgba(99,102,241,.12); border-left: 4px solid #6366f1;
                    font-size: 1.05rem;}
      .block-container {padding-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading CNN encoder and decoder...")
def load_model():
    return ImageCaptioner()


if "history" not in st.session_state:
    st.session_state.history = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

model_ready = MODEL_PATH.exists() and TOKENIZER_PATH.exists()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model status")
    st.write(("✅" if MODEL_PATH.exists() else "❌") + f" `{MODEL_PATH.name}`")
    st.write(("✅" if TOKENIZER_PATH.exists() else "❌") + f" `{TOKENIZER_PATH.name}`")
    if not model_ready:
        st.warning("Trained files are missing, so captions can't be generated yet.")
        st.caption("Put them in `models/` (see `README.md`). Prepare the vocabulary with:")
        st.code("python -m scripts.train", language="bash")

    st.header("Settings")
    override = st.checkbox("Override max caption length", value=False)
    max_len = st.slider("Max length (words)", 5, 60, 34, disabled=not override)
    st.caption("Leave unchecked to use the length the decoder was trained with.")

    st.divider()
    if st.button("Clear history", use_container_width=True, disabled=not st.session_state.history):
        st.session_state.history = []
        st.rerun()

# ------------------------------------------------------------------- main
st.title("🖼️ Image Captioning Classifier")
st.write("Generate natural-language captions for your images with a CNN (InceptionV3) + LSTM model.")

tab_caption, tab_history = st.tabs(["Caption images", f"History ({len(st.session_state.history)})"])

with tab_caption:
    files = st.file_uploader(
        "Upload one or more images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if files:
        images = []
        for f in files:
            try:
                images.append((f.name, Image.open(f).convert("RGB")))
            except Exception:
                st.error(f"Could not open `{f.name}` as an image.")

        run = st.button(
            f"Generate caption{'s' if len(images) != 1 else ''}",
            type="primary",
            disabled=not model_ready or not images,
        )

        results = {}
        if run:
            try:
                captioner = load_model()
                progress = st.progress(0.0, text="Generating captions...")
                for i, (name, img) in enumerate(images):
                    start = time.perf_counter()
                    caption = captioner.predict(img, max_length=max_len if override else None)
                    elapsed = time.perf_counter() - start
                    results[i] = caption
                    st.session_state.history.append(
                        {
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "image": name,
                            "caption": caption,
                            "seconds": round(elapsed, 2),
                        }
                    )
                    progress.progress((i + 1) / len(images), text=f"{i + 1}/{len(images)} done")
                progress.empty()
            except FileNotFoundError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Caption generation failed: {exc}")
                st.info("Check that the model files match this code (same tokenizer, feature size and TensorFlow version).")

        cols_per_row = 3 if len(images) > 1 else 1
        for row_start in range(0, len(images), cols_per_row):
            cols = st.columns(cols_per_row)
            for col, i in zip(cols, range(row_start, min(row_start + cols_per_row, len(images)))):
                name, img = images[i]
                with col:
                    st.image(img, caption=name)
                    if i in results:
                        st.markdown(f'<div class="caption-box">{results[i]}</div>', unsafe_allow_html=True)
                        st.code(results[i], language=None)  # built-in copy button
        if st.button("Remove all images"):
            st.session_state.uploader_key += 1
            st.rerun()
    else:
        st.info("Upload JPG or PNG images to get started.")

with tab_history:
    if not st.session_state.history:
        st.caption("Generated captions will appear here.")
    else:
        st.dataframe(list(reversed(st.session_state.history)), use_container_width=True, hide_index=True)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["time", "image", "caption", "seconds"])
        writer.writeheader()
        writer.writerows(st.session_state.history)
        st.download_button("Download CSV", buf.getvalue(), "captions.csv", "text/csv")
