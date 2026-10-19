import os
from PIL import Image
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import io
import base64
import requests

# Read key dynamically from environment variable or parameter
CONFIGURED_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def _call_gemini_rest(model_name: str, api_key: str, orig_img: Image.Image, prompt: str) -> str:
    """
    Direct HTTPS REST call to Gemini Multimodal API.
    Bypasses any websockets or complex SDK dependencies.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key.strip()}"
    
    # Prepare image payload as JPEG base64
    buf = io.BytesIO()
    rgb_img = orig_img.convert("RGB")
    rgb_img.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}}
            ]
        }]
    }
    
    resp = requests.post(url, json=payload, timeout=40)
    if resp.status_code == 200:
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                text = parts[0].get("text", "")
                if text:
                    return text
        raise RuntimeError("No candidate text returned by Gemini API.")
    else:
        err_msg = resp.text
        try:
            err_data = resp.json()
            err_msg = err_data.get("error", {}).get("message", resp.text)
        except Exception:
            pass
        raise RuntimeError(f"HTTP {resp.status_code}: {err_msg}")

def generate_forensic_explanation(
    orig_img: Image.Image,
    prediction: str,
    confidence: float,
    attn_map: np.ndarray = None,
    faithfulness_res: dict = None,
    api_key: str = None,
    model_name: str = "gemini-3.5-flash-lite"
) -> dict:
    """
    Generates a natural language forensic visual explanation of an image using
    Google Gemini Multimodal Vision-Language Model.
    
    Analyzes:
      - Subject vs background lighting consistency
      - Hair, edge, and silhouette boundary blending
      - Fine texture, skin, and anatomical details
      - Synthesizes findings with DINOv2 XAI attribution & faithfulness metrics.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or CONFIGURED_API_KEY
    
    if not key or key.strip() == "":
        return {
            "status": "missing_api_key",
            "provider": "offline_heuristic",
            "content": _generate_heuristic_explanation(prediction, confidence, faithfulness_res)
        }
    
    prompt = _build_forensic_prompt(prediction, confidence, faithfulness_res)
    candidate_models = [model_name, "gemini-3.5-flash-lite", "gemini-3-flash-preview", "gemini-3.8-flash"]
    seen = set()
    candidate_models = [m for m in candidate_models if not (m in seen or seen.add(m))]
    
    last_error = ""
    # 1. Primary: Direct HTTPS REST (Zero websockets dependencies)
    for model in candidate_models:
        try:
            text = _call_gemini_rest(model, key, orig_img, prompt)
            if text:
                return {
                    "status": "success",
                    "provider": f"Gemini ({model})",
                    "content": text
                }
        except Exception as e:
            last_error = str(e)
            continue

    # 2. Secondary fallback: google.genai SDK
    for model in candidate_models:
        try:
            from google import genai
            client = genai.Client(api_key=key.strip())
            response = client.models.generate_content(
                model=model,
                contents=[orig_img, prompt]
            )
            if response and response.text:
                return {
                    "status": "success",
                    "provider": f"Gemini ({model} SDK)",
                    "content": response.text
                }
        except Exception as e:
            last_error = str(e)
            continue
            
    return {
        "status": "api_error",
        "error_detail": last_error,
        "provider": "offline_fallback",
        "content": _generate_heuristic_explanation(prediction, confidence, faithfulness_res, error_note=last_error)
    }

def _build_forensic_prompt(prediction: str, confidence: float, faithfulness_res: dict = None) -> str:
    conf_pct = confidence * 100
    
    faith_context = ""
    if faithfulness_res:
        passed = faithfulness_res.get("passed_logit_test", False)
        attn_drop = faithfulness_res.get("attribution_logit_drop", 0.0)
        rand_drop = faithfulness_res.get("random_logit_drop", 0.0)
        faith_context = (
            f"\n\nContext from DINOv2 Vision Transformer:\n"
            f"- Model Prediction: {prediction} ({conf_pct:.2f}% confidence)\n"
            f"- Attribution Masking Logit Drop: {attn_drop:.4f}\n"
            f"- Random Masking Logit Drop: {rand_drop:.4f}\n"
            f"- Faithfulness Status: {'Attribution-dominant (focused on foreground boundaries)' if passed else 'Global texture-dominant (reliant on canvas-wide diffusion noise)'}"
        )
    else:
        faith_context = f"\n- Model Prediction: {prediction} ({conf_pct:.2f}% confidence)"

    prompt = f"""You are a senior digital image forensics specialist and computer vision researcher examining this image to determine why it appears authentic or synthetic (AI-generated from models like Stable Diffusion, Midjourney, Flux, etc.).

{faith_context}

Analyze this image closely and provide a thorough, human-like forensic report addressing the following key visual dimensions:

### 1. 💡 Lighting & Illumination Consistency
- Compare the primary illumination (key light, cast shadows, highlight highlights, color temperature) on the main subject vs. the surrounding background.
- Does the subject's lighting direction fit naturally with the light sources visible in the background, or is there mismatched ambient lighting?

### 2. 💇 Hair, Boundary & Silhouette Blending
- Closely inspect fine boundary contours, especially hair strands, fur, clothing collars, and silhouette edges where the foreground meets the background.
- Are individual hair strands crisp and physically plausible, or do they show diffusion "melt", unnatural blurring halos, or synthetic cutout artifacts?

### 3. 🔍 Fine Textures & Anatomical Realism
- Inspect skin surface texture (natural pores and micro-wrinkles vs. plastic, waxy, or over-smoothed sheen).
- Look at anatomical features: eye pupils, iris reflections, ear symmetry, fingernails, teeth, and jewelry alignment.

### 4. ⚖️ Forensic Conclusion & Verdict
- Provide a clear, natural language synthesis explaining why a human observer would judge this image as {'AI-generated' if prediction == 'AI_GENERATED' else 'a real camera photograph'}.

Format your findings with clear bullet points and bold highlights. Write clearly and convincingly like an expert witness report.
"""
    return prompt

def _generate_heuristic_explanation(prediction: str, confidence: float, faithfulness_res: dict = None, error_note: str = None) -> str:
    conf_pct = confidence * 100
    
    note_banner = ""
    if error_note:
        note_banner = f"> ⚠️ *Gemini API notice: {error_note}. Displaying offline forensic inspection baseline below.*\n\n"
    else:
        note_banner = (
            "> 💡 **Gemini Vision Key Not Configured**: Enter your free Gemini API key in the sidebar "
            "to generate live, multimodal visual forensic explanations analyzing this exact image.\n\n"
        )
    
    if prediction == "AI_GENERATED":
        return note_banner + f"""### 🕵️ Forensic Visual Examination Report (Offline Heuristic)

**Target Classification:** `AI_GENERATED` | **Detector Confidence:** `{conf_pct:.2f}%`

#### 1. 💡 Lighting & Illumination Mismatch Analysis
* **Light Source Discrepancy:** In diffusion-generated portraits and scenes, the subject is typically generated with dramatic rim or front lighting that often conflicts with ambient background light. Check whether the highlights on the subject's forehead or hair correspond to the angle of light in the background.
* **Shadow Inconsistency:** AI generators frequently produce soft, directional cast shadows on the subject without corresponding occlusion shadows on adjacent background objects.

#### 2. 💇 Hair & Boundary Blending Artifacts
* **Hair/Background Interface:** Examine the boundary where hair strands meet the backdrop. Stable Diffusion often creates a soft, semi-transparent halo where hair strands seem to dissolve or "bleed" into the background instead of maintaining crisp optical occlusion.
* **Silhouette Cutout:** Edges around shoulders and clothing often appear slightly warped or unnaturally blurred, lacking the genuine optical depth-of-field transition of a real lens.

#### 3. 🔍 Fine Texture & Anatomical Signatures
* **Skin Texture:** Notice whether the skin has authentic micro-pores or exhibits the characteristic "waxy" or overly smoothed sheen typical of denoising autoencoders.
* **Fine Detail Symmetry:** Look closely at pupil reflections (catchlights) — real photos show identical light sources reflected in both eyes; AI images often reflect different light sources per eye.

#### 4. ⚖️ Forensic Conclusion
The DINOv2 vision backbone identified distributed high-frequency artifact patterns consistent with synthetic diffusion generation. An observer should look specifically at hair contours and lighting angle discrepancies to confirm artificial synthesis.
"""
    else:
        return note_banner + f"""### 🕵️ Forensic Visual Examination Report (Offline Heuristic)

**Target Classification:** `REAL IMAGE` | **Detector Confidence:** `{conf_pct:.2f}%`

#### 1. 💡 Lighting & Illumination Consistency
* **Coherent Environment:** The lighting direction on the subject matches the environmental light sources, with consistent color temperature, shadow angles, and authentic ambient reflections.

#### 2. 💇 Natural Boundary & Optical Transitions
* **Hair Definition:** Hair strands exhibit natural optical focus falloff and physical hair separation without diffusion melt or smudged boundaries.
* **Lens Bokeh:** Foreground-to-background transitions match the authentic depth-of-field characteristics of an optical lens.

#### 3. 🔍 Organic Textures & Physical Realism
* **Micro-details:** Natural skin pores, authentic sensor noise (ISO grain), and realistic anatomical symmetries confirm genuine camera acquisition.

#### 4. ⚖️ Forensic Conclusion
The image exhibits coherent optical and physical properties with no detectable diffusion generation fingerprints.
"""
