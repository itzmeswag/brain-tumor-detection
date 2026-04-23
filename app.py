from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model with preprocess_input fix
model = tf.keras.models.load_model(
    "brain_tumor_resnet50_final.keras",
    custom_objects={"preprocess_input": preprocess_input},
    compile=False,
    safe_mode=False
)

# IMPORTANT: Keep same order as training folders
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

def predict_tumor(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    # ResNet50 preprocessing
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction[0])
    confidence = float(np.max(prediction[0]) * 100)

    predicted_class = class_names[predicted_index]

    return predicted_class, confidence


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    img_path = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename != "":
            filename = file.filename
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            result, confidence = predict_tumor(save_path)
            img_path = save_path.replace("\\", "/")

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        img_path=img_path
    )


if __name__ == "__main__":
    app.run(debug=True)
