import streamlit as st
from ultralytics import YOLO
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt

st.set_page_config(page_title="Platelet Abnormality Detector", layout="wide")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model(weight_path):
    return YOLO(weight_path)

st.title("🧬 Platelet Abnormality Detection")

model_path = st.text_input("C:/Users/Jyothi KP/Platelets/Platelets/runs/detect/bccd_platelet_detector7/weights/best.pt", "best.pt")

uploaded_image = st.file_uploader("Upload Blood Smear Image", type=["jpg", "jpeg", "png"])

if uploaded_image and model_path:
    # Load image
    img = Image.open(uploaded_image).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    model = load_model(model_path)

    with st.spinner("Running detection..."):
        results = model(img_cv)

    boxes = results[0].boxes

    if len(boxes) == 0:
        st.warning("No detections found.")
        st.image(img)
        st.stop()

    # ---------------------------------------------
    # Build full detection dataframe
    # ---------------------------------------------
    df_full = pd.DataFrame({
        "xmin": boxes.xyxy[:, 0].cpu().numpy(),
        "ymin": boxes.xyxy[:, 1].cpu().numpy(),
        "xmax": boxes.xyxy[:, 2].cpu().numpy(),
        "ymax": boxes.xyxy[:, 3].cpu().numpy(),
        "confidence": boxes.conf.cpu().numpy(),
        "class": boxes.cls.cpu().numpy(),
    })

    df_full["class_name"] = df_full["class"].apply(lambda x: model.names[int(x)].lower())

    # ---------------------------------------------
    # FILTER ONLY PLATELETS
    # ---------------------------------------------
    df = df_full[df_full["class_name"].str.contains("plate")].reset_index(drop=True)

    st.subheader("🔍 Platelet Detections Table (Filtered)")
    if len(df) == 0:
        st.error("⚠ No platelets detected.")
    else:
        st.dataframe(df)

    # ---------------------------------------------
    # Show annotated image (ALL detections)
    # ---------------------------------------------
    annotated_img = results[0].plot()
    st.subheader("📸 Detection Result (All classes)")
    st.image(annotated_img, channels="BGR")

    # ---------------------------------------------
    # Platelet Shape Analysis (ONLY PLATELETS)
    # ---------------------------------------------
    st.subheader("🧪 Platelet Shape Analysis")

    circularities = []
    abnormal_flags = []
    cropped_images = []

    for i, row in df.iterrows():
        x1, y1, x2, y2 = map(int, [row.xmin, row.ymin, row.xmax, row.ymax])
        crop = img_cv[y1:y2, x1:x2]

        if crop.size == 0:
            circularities.append(np.nan)
            abnormal_flags.append("NA")
            continue

        cropped_images.append(crop)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)

            circularity = 0 if perimeter == 0 else (4 * np.pi * area) / (perimeter ** 2)

            circularities.append(circularity)
            abnormal_flags.append("Abnormal" if circularity < 0.65 else "Normal")

    df["circularity"] = circularities
    df["abnormality"] = abnormal_flags

    st.write(df)

    # ---------------------------------------------
    # Cropped Platelets
    # ---------------------------------------------
    st.subheader("🔬 Cropped Platelets")

    cols = st.columns(4)
    for idx, crop in enumerate(cropped_images):
        with cols[idx % 4]:
            st.image(crop, channels="BGR", caption=f"Platelet {idx+1}")

    # ---------------------------------------------
    # Charts (ONLY platelets)
    # ---------------------------------------------
    st.subheader("📊 Analytics & Charts")

    if len(df) > 0:
        # Confidence histogram
        fig1, ax1 = plt.subplots()
        ax1.hist(df["confidence"], bins=10)
        ax1.set_title("Platelet Detection Confidence")
        ax1.set_xlabel("Confidence")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

        # Normal vs Abnormal count
        fig2, ax2 = plt.subplots()
        count_data = df["abnormality"].value_counts()
        ax2.bar(count_data.index, count_data.values)
        ax2.set_title("Normal vs Abnormal Platelets")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

        # Circularity distribution
        fig3, ax3 = plt.subplots()
        ax3.hist(df["circularity"], bins=10)
        ax3.set_title("Circularity Distribution")
        ax3.set_xlabel("Circularity")
        ax3.set_ylabel("Count")
        st.pyplot(fig3)

    # ---------------------------------------------
    # CSV Download
    # ---------------------------------------------
    csv = df.to_csv(index=False).encode()
    st.download_button("📥 Download Platelet Results CSV", csv, "platelet_results.csv", "text/csv")
