# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:5173",            # 本地开发
    "https://<你的前端域名>.vercel.app"  # 上线后替换成真实域名
]}})
import os, time
from model import TinyNN
from utils import preprocess_pixels


net = TinyNN()
if os.path.exists("model.npz"):
    net.load("model.npz")
    print("Loaded model.npz")
else:
    print("model.npz not found. 使用随机权重（几乎识别不了）。先运行 train.py 训练一下。")

@app.route("/hello")
def hello():
    return jsonify({"ok": True, "msg": "Hello from Flask!"})

@app.route("/upload", methods=["POST"])
def upload():
    t0 = time.time()
    data = request.get_json(force=True)
    pixels = data.get("pixels")
    if pixels is None or len(pixels) != 28*28:
        return jsonify({"ok": False, "error": "Expected 'pixels' length 784"}), 400
    X = preprocess_pixels(pixels)
    pred, probs = net.predict(X)
    return jsonify({
        "ok": True, "pred": int(pred[0]),
        "probs": probs[0].tolist(),
        "time_ms": int((time.time()-t0)*1000)
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render 会注入 PORT
    app.run(host="0.0.0.0", port=port, debug=False)
