from flask import Blueprint, request, jsonify, send_file
import tensorflow as tf
tf.config.run_functions_eagerly(False)

import numpy as np
from PIL import Image
import io
import os
import json
import zipfile
import time
import traceback

api_bp = Blueprint("api", __name__)

import util
config = util.init_config("./config.json")
TARGET_SIZE = tuple(config["TARGET_SIZE"])
MODELS = config["MODELS"]
MODEL_DIR = "./models"

model_results_cache = {}
loaded_models = {}  # Lazy-loaded models

def load_results(full_model_name):
    if full_model_name in model_results_cache:
        return model_results_cache[full_model_name]

    results_path = os.path.join(MODEL_DIR, f"{full_model_name}_results.json")
    results = {"stats": {"per_class": []}}
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)

    model_results_cache[full_model_name] = results
    print(f"[INFO] Loaded results for {full_model_name}")
    return results

def load_model(full_model_name):
    if full_model_name in loaded_models:
        return loaded_models[full_model_name]

    keras_path = os.path.join(MODEL_DIR, f"{full_model_name}.keras")
    if not os.path.exists(keras_path):
        print(f"[WARN] Model not found: {full_model_name}")
        return None

    print(f"[INFO] Loading model: {full_model_name} ...")
    start_time = time.time()
    model = tf.keras.models.load_model(
        keras_path,
        custom_objects={
            "WeightedLoss": util.WeightedLoss,
            "WeightedAccuracy": util.WeightedAccuracy,
            "NormalAccuracy": util.NormalAccuracy,
            "RandomSegmentation": util.RandomSegmentation,
            "ResizeLike": util.ResizeLike
        },
        safe_mode=False
    )
    loaded_models[full_model_name] = model
    print(f"[INFO] Model {full_model_name} loaded in {time.time() - start_time:.2f}s")
    return model

def run_segmentation(image_bytes, full_model_name):
    start_time = time.time()
    try:
        # Decode and resize for model
        image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, TARGET_SIZE)
        image = image[None, ...]  # Add batch dimension

        # Prepare original resized PNG
        pil_original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_original = pil_original.resize((TARGET_SIZE[1], TARGET_SIZE[0]))
        orig_buf = io.BytesIO()
        pil_original.save(orig_buf, format="PNG")
        orig_buf.seek(0)
        orig_png_bytes = orig_buf.getvalue()

        # Load model lazily
        model = load_model(full_model_name)
        if model is None:
            raise RuntimeError(f"Model not loaded: {full_model_name}")

        # Predict
        pred_start = time.time()
        prediction = model(image, training=False).numpy()[0]
        print(f"[INFO] Prediction for {full_model_name} took {time.time() - pred_start:.2f}s")

        pred_classes = prediction.argmax(axis=-1).astype(np.uint8)
        palette = np.array(util.get_pallette(config["CLASSES"]), dtype=np.uint8)
        colored_mask = palette[pred_classes]

        mask_pil = Image.fromarray(colored_mask)
        mask_buf = io.BytesIO()
        mask_pil.save(mask_buf, format="PNG")
        mask_buf.seek(0)
        mask_png_bytes = mask_buf.getvalue()

        # Compute legend
        results = load_results(full_model_name)["stats"]
        total_pixels = pred_classes.size
        mean_iou = float(np.mean([c["iou"] for c in results["per_class"]]))

        legend = []
        for class_index, c in enumerate(results["per_class"]):
            fraction = np.sum(pred_classes == class_index) / total_pixels
            legend.append({
                "name": c["name"], "color": c["color"],
                "tp": c["TP"], "tn": c["TN"], "fp": c["FP"], "fn": c["FN"],
                "iou": c["iou"], "precision": c["precision"], "recall": c["recall"],
                "f1_score": c["f1_score"], "accuracy": c["accuracy"],
                "fraction": fraction, "mean_iou": mean_iou
            })

        legend_json_str = json.dumps(legend, indent=2)
        print(f"[INFO] run_segmentation {full_model_name} done in {time.time() - start_time:.2f}s")
        return orig_png_bytes, mask_png_bytes, legend_json_str

    except Exception:
        print(f"[ERROR] run_segmentation failed for {full_model_name}")
        traceback.print_exc()
        raise

@api_bp.route("/predict_all", methods=["POST"])
def predict_all():
    request_start = time.time()
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded."}), 400

        img_bytes = request.files["image"].read()
        print(f"[INFO] Received image of size {len(img_bytes)} bytes")

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            for short_name, model_data in MODELS.items():
                full_name = model_data["full_name"]
                description = model_data.get("description", {})
                print(f"[INFO] Processing model {full_name} ...")
                orig_png, mask_png, legend_json = run_segmentation(img_bytes, full_name)

                zf.writestr(f"orig-{short_name}.png", orig_png)
                zf.writestr(f"mask-{short_name}.png", mask_png)
                zf.writestr(f"legend-{short_name}.json", legend_json)
                zf.writestr(f"description-{short_name}.json", json.dumps(description, indent=2))
                print(f"[INFO] Finished model {full_name}")

        zip_buf.seek(0)
        print(f"[INFO] /predict_all finished in {time.time() - request_start:.2f}s")
        return send_file(
            zip_buf,
            mimetype="application/zip",
            as_attachment=True,
            download_name="prediction.zip"
        )

    except Exception as e:
        print(f"[ERROR] /predict_all failed after {time.time() - request_start:.2f}s")
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
