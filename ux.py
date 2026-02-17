import sys
import os
import streamlit as st
from streamlit_folium import st_folium
import folium
from PIL import Image
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "notebooks"))
from model import GeoModel, device

import geoguessr_inference as gi

WEIGHTS_PATH = os.environ.get("BRIDGE_WEIGHTS", "alphabridge.pt")

st.set_page_config(page_title="GeoGuessr Bot", layout="wide")
st.title("GeoGuessr Bot")


@st.cache_resource
def load_model():
    """Load StreetCLIP + AlphaBridge (frozen CLIP, trained bridge)."""
    model = GeoModel().to(device)
    if os.path.exists(WEIGHTS_PATH):
        model.bridge.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        model.eval()
    else:
        st.warning(
            f"No bridge weights found at `{WEIGHTS_PATH}`. "
            "Set BRIDGE_WEIGHTS env var or place alphabridge.pt in the project root. "
            "Running with untrained bridge."
        )
        model.eval()
    return model


model = load_model()


def geolocate(image: Image.Image) -> gi.InferenceResult:
    """PIL image -> InferenceResult (weighted lat/lon + top-3 predictions)."""
    clip_vec = model.encode_images([image])           # (1, 768)
    with torch.no_grad():
        alpha_vec = model(clip_vec)                    # (1, 64)
    query = alpha_vec.squeeze(0).cpu().tolist()         # list[float] len 64
    return gi.infer_location(query)


# ── Sidebar: image upload ─────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Image")
    uploaded = st.file_uploader("Street-level photo", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────
if uploaded:
    with st.spinner("Geolocating..."):
        result = geolocate(img)

    col_map, col_info = st.columns([2, 1])

    with col_map:
        m = folium.Map(location=[result.weighted_lat, result.weighted_lon], zoom_start=5)

        # Main prediction (weighted)
        folium.Marker(
            [result.weighted_lat, result.weighted_lon],
            popup=f"Weighted: {result.weighted_lat:.4f}, {result.weighted_lon:.4f}",
            icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
        ).add_to(m)

        # Top-k candidate circles scaled by confidence
        colors = ["blue", "green", "orange"]
        for i, pred in enumerate(result.predictions):
            folium.CircleMarker(
                [pred.lat, pred.lon],
                radius=max(pred.confidence_pct * 0.3, 5),
                color=colors[i % len(colors)],
                fill=True,
                fill_opacity=0.5,
                popup=f"#{i+1}: ({pred.lat:.4f}, {pred.lon:.4f}) — {pred.confidence_pct:.1f}%",
            ).add_to(m)

        st_folium(m, width=700, height=500, returned_objects=[])

    with col_info:
        st.subheader("Prediction")
        st.metric("Weighted Location", f"{result.weighted_lat:.4f}, {result.weighted_lon:.4f}")

        st.divider()
        st.subheader("Candidates")
        for i, pred in enumerate(result.predictions):
            st.markdown(
                f"**#{i+1}** — `{pred.lat:.4f}, {pred.lon:.4f}`  \n"
                f"Confidence: **{pred.confidence_pct:.1f}%** · S2: `{pred.s2_token}`"
            )
else:
    m = folium.Map(location=[20, 0], zoom_start=2)
    st_folium(m, width=900, height=500, returned_objects=[])
    st.info("Upload a street-level image to get started.")
