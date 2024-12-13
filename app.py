import os
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from PIL import Image
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from generator_model import Generator

# Flask initialisation
app = Flask(__name__)

# folder to save the images
UPLOAD_FOLDER = "C:/Idris/Ecole/2A/CycleGAN/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load the generator
checkpoint_path = "C:/Idris/Ecole/2A/CycleGAN/genc.pth.tar"
gen_C_loaded = Generator(img_channels=3, num_residuals=9).to(DEVICE)
gen_C_loaded.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)["state_dict"])
gen_C_loaded.eval()

# transformation of the image
transform = A.Compose([
    A.Resize(width=256, height=256),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    augmented = transform(image=image)
    tensor_image = augmented["image"].unsqueeze(0)
    return tensor_image.to(DEVICE)

def postprocess_image(tensor_image):
    tensor_image = tensor_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    tensor_image = (tensor_image + 1) / 2
    tensor_image = (tensor_image * 255).astype(np.uint8)
    return Image.fromarray(tensor_image)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded.", 400
        file = request.files["file"]
        if file.filename == "":
            return "No file selected.", 400

        # save the original image
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)
        input_tensor = preprocess_image(file_path)

        # generate the new image
        with torch.no_grad():
            generated_tensor = gen_C_loaded(input_tensor)

        # convert and save the generated image
        output_filename = "generated_" + file.filename
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)
        generated_image = postprocess_image(generated_tensor)
        generated_image.save(output_path)

        # redirection to the display page
        return redirect(url_for("display", filename=output_filename))

    # front page
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/display/<filename>")
def display(filename):
    # URL of the generated image
    image_url = url_for("uploaded_file", filename=filename)
    return render_template("display.html", image_path=image_url)

if __name__ == "__main__":
    app.run(debug=True)
