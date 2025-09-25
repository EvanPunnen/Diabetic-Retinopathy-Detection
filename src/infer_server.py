# src/infer_server.py
import os
from flask import Flask, request, render_template, send_file, redirect, url_for
import torch
from torchvision import transforms
from PIL import Image
from modelzoo import build_classifier, load_checkpoint
from utils import clahe_enhance_pil, map_label_to_text, generate_pdf_report, ICDRS_LABELS

app = Flask(__name__, template_folder="templates")
MODEL_PATH = "models/densenet121_best.pth"  # update if different
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load model
model = build_classifier("densenet121", num_classes=5, pretrained=False)
model, _ = load_checkpoint(MODEL_PATH, model, device=DEVICE)
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("fundus")
    patient = request.form.get("patient", "Anonymous")
    dob = request.form.get("dob", "")
    if not file:
        return redirect(url_for("index"))

    save_dir = "outputs"
    os.makedirs(save_dir, exist_ok=True)
    img_path = os.path.join(save_dir, file.filename)
    file.save(img_path)

    # preprocess (CLAHE for contrast)
    pil = Image.open(img_path).convert("RGB")
    pil = clahe_enhance_pil(pil)
    inp = transform(pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
        pred_idx = int(probs.argmax())

    pred_text = map_label_to_text(pred_idx)

    # generate PDF report
    pdf_path = os.path.join(save_dir, f"report_{os.path.splitext(file.filename)[0]}.pdf")
    generate_pdf_report(pdf_path, patient, dob, img_path, pred_text, probs.tolist())

    return render_template("result.html",
                           patient=patient,
                           dob=dob,
                           image_filename=file.filename,
                           prediction=pred_text,
                           probs=list(zip(range(5), [f"{p:.3f}" for p in probs])),
                           pdf_path=pdf_path)

@app.route("/download/<path:pdf_name>")
def download(pdf_name):
    return send_file(pdf_name, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
