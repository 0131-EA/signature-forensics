import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# --- Configuration & Custom Styling ---
st.set_page_config(page_title="Signature Intelligence Pro", layout="wide")

# Applying a professional forensic aesthetic
st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    .stMetric { border: 2px solid #2c3e50; border-radius: 10px; padding: 20px; background: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

st.title("🖋️ Signature Forensic Intelligence")
st.write("Professional-grade tool for structural and density-based signature verification.")

# --- Logic Modules ---

def clean_forensic_image(img):
    """Performs noise reduction and advanced binarization."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Removing paper texture while keeping ink edges sharp
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    # Automatically finding the best threshold for the ink
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def get_grid_density(thresh, grid_size=10):
    """Divides the signature into 100 cells to analyze local ink distribution."""
    h, w = thresh.shape
    dy, dx = h // grid_size, w // grid_size
    # Measuring the percentage of ink in each grid cell
    densities = [np.sum(thresh[i*dy:(i+1)*dy, j*dx:(j+1)*dx] == 255) / (dy * dx) 
                 for i in range(grid_size) for j in range(grid_size)]
    return np.array(densities)

def analyze_signatures(ref_img, test_img):
    """Compares Global Structure and Local Density."""
    # Standardizing the test image size to match the reference
    test_res = cv2.resize(test_img, (ref_img.shape[1], ref_img.shape[0]))
    
    g1 = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(test_res, cv2.COLOR_BGR2GRAY)
    
    # 1. Global structural check (Shape)
    global_score, _ = ssim(g1, g2, full=True)
    
    # 2. Local density check (Pressure/Weight)
    d1 = get_grid_density(clean_forensic_image(ref_img))
    d2 = get_grid_density(clean_forensic_image(test_res))
    local_score = np.corrcoef(d1, d2)[0, 1]
    
    # 3. Generating the discrepancy map
    diff = cv2.absdiff(g1, g2)
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    
    return global_score, local_score, heatmap

# --- UI Flow ---

st.info("Upload a known 'Reference' signature and a 'Questioned' sample to verify authenticity.")

col_a, col_b = st.columns(2)
with col_a:
    ref_file = st.file_uploader("Reference Signature (Genuine)", type=["png", "jpg", "jpeg"], key="ref")
with col_b:
    test_file = st.file_uploader("Questioned Signature (Test)", type=["png", "jpg", "jpeg"], key="test")

if ref_file and test_file:
    # Converting uploaded files to OpenCV format
    img_ref = cv2.imdecode(np.frombuffer(ref_file.read(), np.uint8), 1)
    img_test = cv2.imdecode(np.frombuffer(test_file.read(), np.uint8), 1)
    
    g_score, l_score, h_map = analyze_signatures(img_ref, img_test)
    
    # Final probability calculation (60% weight on shape, 40% on density distribution)
    final_score = (g_score * 0.6) + (l_score * 0.4)
    
    st.divider()
    
    # Visual Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Structural Match", f"{int(g_score * 100)}%")
    m2.metric("Density Alignment", f"{int(l_score * 100)}%")
    m3.metric("Trust Probability", f"{int(final_score * 100)}%")
    
    if final_score > 0.85:
        st.success("✅ VERDICT: VERIFIED GENUINE. High consistency in both shape and ink distribution.")
    elif final_score > 0.65:
        st.warning("⚠️ VERDICT: INCONCLUSIVE. Significant variations detected. Manual review suggested.")
    else:
        st.error("❌ VERDICT: POSSIBLE FORGERY. The structural and density patterns do not match the reference.")

    st.subheader("Discrepancy Analysis Map")
    st.write("Red and yellow areas highlight where the signature paths differ from the original.")
    st.image(h_map, caption="Structural Deviation Heatmap", channels="BGR", use_column_width=True)