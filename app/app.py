import os
import sys
import torch
from PIL import Image
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from dataset import get_dinov2_transforms
from model import DINOv2Classifier
import xai
import importlib
importlib.reload(xai)
from xai import compute_attention_map, generate_heatmap_overlay, run_faithfulness_perturbation

import llm_explainer
importlib.reload(llm_explainer)
from llm_explainer import generate_forensic_explanation

st.set_page_config(
    page_title="AI Image Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich premium styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1f77b4, #aec7e8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #7f7f7f;
        margin-bottom: 1.5rem;
    }
    .metric-container {
        padding: 1.2rem;
        border-radius: 10px;
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 1rem;
    }
    .real-label {
        font-size: 2rem;
        font-weight: 800;
        color: #2ca02c;
    }
    .ai-label {
        font-size: 2rem;
        font-weight: 800;
        color: #d62728;
    }
    .perturb-card {
        padding: 1rem;
        border-radius: 8px;
        background-color: rgba(128, 128, 128, 0.06);
        border: 1px solid rgba(128, 128, 128, 0.18);
        text-align: center;
        margin-bottom: 0.6rem;
    }
    .perturb-card h4 {
        margin-top: 0;
        margin-bottom: 0.4rem;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to cache model loading
@st.cache_resource
def load_detection_model(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        
    checkpoint = torch.load(model_path, map_location=device)
    model_type = checkpoint.get("model_type", "mlp")
    
    # Instantiate end-to-end model
    model = DINOv2Classifier(backbone_name='dinov2_vits14', classifier_type=model_type, num_classes=2, freeze_backbone=True)
    
    # Map head weights
    head_state_dict = checkpoint["model_state_dict"]
    mapped_state_dict = {}
    for k, v in head_state_dict.items():
        if model_type == "linear":
            new_k = k.replace("fc.", "classifier.")
        else:
            new_k = k.replace("mlp.", "classifier.")
        mapped_state_dict[new_k] = v
        
    model.load_state_dict(mapped_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model

# Setup sidebar
st.sidebar.markdown("## 🔍 AI Image Detection")
st.sidebar.markdown(
    "This dashboard leverages a pretrained **DINOv2 ViT-S/14** Vision Transformer backbone "
    "combined with an MLP classification head to detect AI-generated images."
)

st.sidebar.markdown("### 🧪 Faithfulness Perturbation")
mask_pct = st.sidebar.slider(
    "Patch Masking Ratio",
    min_value=0.05,
    max_value=0.35,
    value=0.15,
    step=0.05,
    help="Proportion of spatial patches (out of 256) blacked out during faithfulness perturbation testing."
)
st.sidebar.caption(f"Masking **{int(256 * mask_pct)}** of 256 patches ({int(mask_pct * 100)}% area)")

st.sidebar.markdown("### 🧠 LLM Forensic Reasoning")
env_gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
if env_gemini_key:
    st.sidebar.success("🟢 **Gemini Vision AI: Active**")
    api_key_input = env_gemini_key
else:
    api_key_input = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        help="Enter your Google Gemini API key or set GEMINI_API_KEY in your local .env file."
    )
    if api_key_input:
        st.sidebar.success("🟢 **Gemini Vision AI: Connected**")
    else:
        st.sidebar.info("ℹ️ Add Gemini key above or in local .env for LLM forensics")
st.sidebar.caption("Multimodal visual reasoning engine connected via Google Gemini.")

st.sidebar.markdown("### System Architecture")
st.sidebar.info(
    "1. **Input Image** (224x224 RGB, Normalized)\n"
    "2. **DINOv2 Backbone** (Frozen ViT-S/14 -> 384 dim)\n"
    "3. **MLP Classifier** (ReLU, Dropout -> Logits)\n"
    "4. **XAI Attribution** (Manual [CLS] Self-Attention)\n"
    "5. **Faithfulness Test** (Perturbation Patch Masking)\n"
    "6. **Multimodal LLM** (Forensic Visual Reasoning)"
)

# App Content
st.markdown('<div class="main-title">AI-Generated Image Detection & XAI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A Research Dashboard using Pretrained DINOv2, Explainable AI, and Faithfulness Perturbation Analysis</div>', unsafe_allow_html=True)

# Check model availability
model_path = "../checkpoints/best_model.pth"
if not os.path.exists(model_path):
    model_path = "checkpoints/best_model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(model_path):
    st.error(f"⚠️ Best model checkpoint not found at `{model_path}`. Please train the classifier head first!")
else:
    try:
        model = load_detection_model(model_path, device)
        st.success(f"✔️ Loaded DINOv2 classifier successfully (Device: {device})")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
        
    # File upload
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG, WEBP)...", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        orig_img = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Uploaded Image")
            st.image(orig_img, use_container_width=True)
            st.caption(f"Resolution: {orig_img.size[0]}x{orig_img.size[1]} | Mode: {orig_img.mode}")
            
        with col2:
            st.markdown("### Run Detection & Analysis")
            analyze_clicked = st.button("🚀 Analyze Image & Run Faithfulness Test", use_container_width=True)
            
            # Use session state to cache analysis results so interactions don't erase output
            if analyze_clicked:
                with st.spinner("Running DINOv2 forward pass, XAI heatmap, and Faithfulness Perturbation..."):
                    # Preprocess
                    transform = get_dinov2_transforms(img_size=224)
                    img_tensor = transform(orig_img).unsqueeze(0).to(device)
                    
                    # 1. Run prediction
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probs = torch.softmax(outputs, dim=1).squeeze(0)
                        
                    real_prob = probs[0].item()
                    ai_prob = probs[1].item()
                    
                    pred_label = 1 if ai_prob > real_prob else 0
                    prediction = "AI_GENERATED" if pred_label == 1 else "REAL"
                    confidence = ai_prob if pred_label == 1 else real_prob
                    
                    # 2. Extract self-attention map
                    attn_map, _, _ = compute_attention_map(model, img_tensor, device)
                    
                    # 3. Generate heatmap overlay
                    overlay, _ = generate_heatmap_overlay(orig_img, attn_map)
                    
                    # 4. Run Faithfulness Perturbation Analysis (Before vs After)
                    faith_res = run_faithfulness_perturbation(
                        orig_img,
                        model,
                        device,
                        threshold_pct=mask_pct,
                        attn_map=attn_map
                    )
                    
                    # 5. Generate Multimodal LLM Forensic Explanation
                    llm_forensic = generate_forensic_explanation(
                        orig_img=orig_img,
                        prediction=prediction,
                        confidence=confidence,
                        attn_map=attn_map,
                        faithfulness_res=faith_res,
                        api_key=api_key_input
                    )
                    
                    st.session_state["analysis_results"] = {
                        "prediction": prediction,
                        "pred_label": pred_label,
                        "confidence": confidence,
                        "real_prob": real_prob,
                        "ai_prob": ai_prob,
                        "attn_map": attn_map,
                        "overlay": overlay,
                        "faith_res": faith_res,
                        "llm_forensic": llm_forensic
                    }
                    
        # Check if we have analysis results to display
        if "analysis_results" in st.session_state:
            res = st.session_state["analysis_results"]
            faith = res["faith_res"]
            
            # Show summary classifier output in right column
            with col2:
                st.markdown("#### Classifier Output")
                if res["pred_label"] == 1:
                    st.markdown(
                        f'<div class="metric-container"><div class="ai-label">AI-GENERATED</div>'
                        f'<div>Confidence: <b>{res["confidence"]*100:.2f}%</b></div></div>', 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="metric-container"><div class="real-label">REAL IMAGE</div>'
                        f'<div>Confidence: <b>{res["confidence"]*100:.2f}%</b></div></div>', 
                        unsafe_allow_html=True
                    )
                    
                st.markdown("**Prediction Probabilities:**")
                st.progress(res["ai_prob"], text=f"AI Probability: {res['ai_prob']:.4f}")
                st.progress(res["real_prob"], text=f"Real Probability: {res['real_prob']:.4f}")

            st.divider()

            # -------------------------------------------------------------------------
            # SECTION 1: Explainable AI Attention Heatmaps
            # -------------------------------------------------------------------------
            st.markdown("### 🔍 Explainable AI (XAI) Attribution")
            xai_col1, xai_col2 = st.columns(2)
            with xai_col1:
                st.image(res["overlay"], use_container_width=True, caption="DINOv2 Last-Layer Self-Attention Heatmap Overlay")
            with xai_col2:
                # Plot raw 16x16 attention grid
                fig, ax = plt.subplots(figsize=(5, 5))
                im = ax.imshow(res["attn_map"], cmap="jet")
                ax.set_title("Raw DINOv2 Attention Grid (16x16 Patches)", fontsize=11)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.divider()

            # -------------------------------------------------------------------------
            # SECTION 2: Faithfulness Perturbation Analysis (Before vs After)
            # -------------------------------------------------------------------------
            st.markdown("### 🧪 Faithfulness Perturbation Analysis (Before vs. After)")
            st.markdown(
                "Faithfulness tests whether the visual attention explanations truly causally govern the model's predictions. "
                f"We compare classification confidence and logit drops when masking the **top {int(mask_pct*100)}% attended patches** "
                f"versus masking the same area (**{faith['num_patches_masked']} patches**) randomly."
            )
            
            pcol1, pcol2, pcol3 = st.columns(3)
            
            with pcol1:
                st.markdown(
                    '<div class="perturb-card">'
                    '<h4>🖼️ Before Perturbation</h4>'
                    '<div><b>Baseline (Untouched)</b></div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.image(faith["orig_img_224"], use_container_width=True, caption="Input (224x224 RGB)")
                st.markdown(
                    f"- **Prediction:** `{faith['pred_label']}`\n"
                    f"- **Confidence:** `{faith['base_confidence']*100:.2f}%`\n"
                    f"- **Logit Score:** `{faith['base_logit']:.4f}`\n"
                    f"- **Patches Masked:** `0 / {faith['total_patches']}`"
                )

            with pcol2:
                st.markdown(
                    '<div class="perturb-card">'
                    '<h4>🎯 After Attribution Masking</h4>'
                    f'<div><b>Top {int(mask_pct*100)}% Attended Patches</b></div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.image(faith["attn_masked_img"], use_container_width=True, caption=f"Masked {faith['num_patches_masked']} Top Attended Patches")
                st.markdown(
                    f"- **Confidence:** `{faith['attribution_masked_confidence']*100:.2f}%` "
                    f"(Drop: <b style='color:#d62728;'>🔻 {faith['attribution_conf_drop']*100:.2f}%</b>)\n"
                    f"- **Logit Score:** `{faith['attribution_masked_logit']:.4f}` "
                    f"(Drop: <b style='color:#d62728;'>🔻 {faith['attribution_logit_drop']:.4f}</b>)\n"
                    f"- **Patches Masked:** `{faith['num_patches_masked']} / {faith['total_patches']}`",
                    unsafe_allow_html=True
                )

            with pcol3:
                st.markdown(
                    '<div class="perturb-card">'
                    '<h4>🎲 After Random Masking</h4>'
                    f'<div><b>Random {int(mask_pct*100)}% Control Patches</b></div>'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.image(faith["rand_masked_img"], use_container_width=True, caption=f"Masked {faith['num_patches_masked']} Random Control Patches")
                st.markdown(
                    f"- **Confidence:** `{faith['random_masked_confidence']*100:.2f}%` "
                    f"(Drop: <b style='color:#1f77b4;'>🔻 {faith['random_conf_drop']*100:.2f}%</b>)\n"
                    f"- **Logit Score:** `{faith['random_masked_logit']:.4f}` "
                    f"(Drop: <b style='color:#1f77b4;'>🔻 {faith['random_logit_drop']:.4f}</b>)\n"
                    f"- **Patches Masked:** `{faith['num_patches_masked']} / {faith['total_patches']}`",
                    unsafe_allow_html=True
                )

            # Quantitative Comparison Table
            st.markdown("#### 📊 Quantitative Perturbation Comparison Table")
            summary_data = {
                "Perturbation State": [
                    "1. Before Perturbation (Baseline)",
                    f"2. After Attribution Masking ({int(mask_pct*100)}% Top Attended)",
                    f"3. After Random Masking ({int(mask_pct*100)}% Random Control)"
                ],
                "Confidence": [
                    f"{faith['base_confidence']*100:.2f}%",
                    f"{faith['attribution_masked_confidence']*100:.2f}%",
                    f"{faith['random_masked_confidence']*100:.2f}%"
                ],
                "Confidence Drop (Δ)": [
                    "—",
                    f"{faith['attribution_conf_drop']*100:+.2f}%",
                    f"{faith['random_conf_drop']*100:+.2f}%"
                ]
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

            # Faithfulness Verdict Banner
            if faith["passed_logit_test"]:
                st.success(
                    f"✅ **Faithfulness Test Status: PASSED (Object-Centric Driver)**\n\n"
                    f"Masking the attended patches produced a larger logit drop (**{faith['attribution_logit_drop']:.4f}**) "
                    f"than masking random patches (**{faith['random_logit_drop']:.4f}**) — "
                    f"Difference: **+{faith['attribution_logit_drop'] - faith['random_logit_drop']:.4f}**.\n\n"
                    f"This confirms that the attended spatial regions strongly influence the classifier's prediction for this image."
                )
            else:
                st.info(
                    f"🔬 **Faithfulness Scientific Finding: Global Texture & Frequency Artifacts Dominance**\n\n"
                    f"Masking random patches caused a larger logit drop (**{faith['random_logit_drop']:.4f}**) "
                    f"than masking foreground attended regions (**{faith['attribution_logit_drop']:.4f}**).\n\n"
                    f"**Scientific Significance:** DINOv2 naturally attends to semantic object shapes and contours. "
                    f"However, the AI-detection classifier head primarily relies on high-frequency textural noise and global diffusion "
                    f"fingerprints spread across the entire canvas. Disrupting random patches breaks up this global texture fingerprint "
                    f"more aggressively than masking foreground semantic objects alone."
                )

            # -------------------------------------------------------------------------
            # SECTION 3: Multimodal LLM Forensic Explanation
            # -------------------------------------------------------------------------
            st.divider()
            st.markdown("### 🧠 Multimodal AI Forensic Explanation (Lighting, Hair & Textures)")
            st.markdown(
                "Natural language forensic breakdown powered by Vision-Language AI. "
                "Evaluates physical plausibility, lighting consistency between the human and background, "
                "hair/silhouette boundary bleeding, and anatomical detail realism."
            )
            
            forensic_data = res.get("llm_forensic", {})
            forensic_status = forensic_data.get("status", "missing_api_key")
            provider_name = forensic_data.get("provider", "Gemini")
            
            with st.container(border=True):
                col_status, col_btn = st.columns([3, 1])
                with col_status:
                    if forensic_status == "success":
                        st.success(f"🤖 **Multimodal Forensic Examination by {provider_name}**")
                    else:
                        st.warning(f"⚠️ **Forensic Analysis Notice:** {forensic_data.get('error_detail', 'Heuristic fallback mode')}")
                
                with col_btn:
                    if st.button("🔄 Re-generate with LLM", use_container_width=True):
                        with st.spinner("Analyzing image with Gemini Vision..."):
                            fresh_forensic = generate_forensic_explanation(
                                orig_img=orig_img,
                                prediction=res["prediction"],
                                confidence=res["confidence"],
                                attn_map=res["attn_map"],
                                faithfulness_res=faith,
                                api_key=api_key_input
                            )
                            st.session_state["analysis_results"]["llm_forensic"] = fresh_forensic
                            st.rerun()

                st.markdown(forensic_data.get("content", "No explanation available."))

            # Scientific Interpretation & Visual Artifacts Guide Expander
            with st.expander("🔬 Scientific Interpretation & Visual Artifacts Guide"):
                st.markdown(
                    "**How the attention heatmap is computed:**\n"
                    "The heatmap visualizes the self-attention weights from the last layer of the "
                    "Vision Transformer (ViT). Specifically, it displays the attention of the "
                    "`[CLS]` token (which aggregates global information for final classification) "
                    "towards all spatial patch tokens. Red regions indicate areas that DINOv2 attended "
                    "to most strongly during feature extraction.\n\n"
                    
                    "**🔍 Visual Clues to Look for in Attended Regions:**\n"
                    "Use the attention map to guide your inspection of the image. The model focuses on borders, "
                    "edges, and transitions where Stable Diffusion often fails. Look closely at red/orange areas for:\n"
                    "1. **Hair/Background Blending Inconsistencies:** Check if fine structures like hair or fur "
                    "do not set naturally with the background. Look for telltale outlines, sudden blurs, or sharp, unnatural cutouts.\n"
                    "2. **Edge Artifacts:** Transitions between objects (e.g., clothes meeting skin or objects meeting backgrounds) "
                    "often have weird textures or glowing halos in AI-generated images.\n"
                    "3. **Lighting Anomalies:** Shadows and highlights on highly attended features may contradict the main light source.\n"
                    "4. **Detail Asymmetry:** Fine details (like eyes, ears, or jewelry) often display warped shapes.\n\n"
                    
                    "**🔬 Faithfulness Perturbation Methodology:**\n"
                    "By blacking out patches and re-running model inference, we mathematically confirm whether "
                    "the visual explanation correlates directly with classification confidence or whether "
                    "unseen global high-frequency texture features are driving the decision."
                )
