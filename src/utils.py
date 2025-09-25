# src/utils.py
import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import os

ICDRS_LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative"
}

def clahe_enhance_pil(pil_img):
    # PIL -> OpenCV CLAHE -> PIL
    img = np.array(pil_img)[:,:,::-1]  # RGB->BGR
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl,a,b))
    bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    rgb = bgr[:,:,::-1]
    return Image.fromarray(rgb)

def map_label_to_text(label_idx):
    return ICDRS_LABELS.get(int(label_idx), "Unknown")

def generate_pdf_report(out_path, patient_name, dob, image_path, predicted_label, probabilities):
    """
    Generates a simple one-page PDF report.
    probabilities: list or dict of probabilities per class (0..4)
    """
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height-60, "Diabetic Retinopathy Screening Report")
    c.setFont("Helvetica", 11)
    c.drawString(40, height-90, f"Patient: {patient_name}")
    c.drawString(40, height-110, f"DOB: {dob}")
    c.drawString(40, height-130, f"Predicted DR Level: {predicted_label}")
    c.drawString(40, height-150, "Probabilities:")
    y = height-170
    for i,p in enumerate(probabilities):
        c.drawString(60, y, f"{i} - {ICDRS_LABELS.get(i)} : {p:.3f}")
        y -= 16

    # add image (shrink to fit)
    try:
        max_w = 300
        max_h = 300
        from PIL import Image
        img = Image.open(image_path)
        iw, ih = img.size
        scale = min(max_w/iw, max_h/ih, 1.0)
        img = img.resize((int(iw*scale), int(ih*scale)))
        tmp = os.path.join(os.path.dirname(out_path), "tmp_report_img.jpg")
        img.save(tmp)
        c.drawImage(tmp, width- max_w - 40, height - max_h - 40, width=int(iw*scale), height=int(ih*scale))
        if os.path.exists(tmp):
            os.remove(tmp)
    except Exception as e:
        print("Warning: could not embed image in PDF:", e)

    c.showPage()
    c.save()
