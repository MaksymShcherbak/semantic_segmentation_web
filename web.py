from flask import Blueprint, render_template
from api import MODELS, loaded_models, MODEL_DIR
import os

web_bp = Blueprint("web", __name__)

@web_bp.route("/", methods=["GET"])
def index():
    loaded_short_names = []

    for short_name, model_data in MODELS.items():
        full_name = model_data["full_name"]
        if full_name in loaded_models:
            loaded_short_names.append(short_name)

    return render_template("index.html", models=loaded_short_names)
