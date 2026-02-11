import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="

from flask import Flask
from api import api_bp
from web import web_bp

app = Flask(__name__)

app.register_blueprint(api_bp, url_prefix="/api")
app.register_blueprint(web_bp, url_prefix="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)