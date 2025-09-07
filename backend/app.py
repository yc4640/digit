# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# 调试期可用通配放行（最简单，先确认能通）
# CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

# 线上建议白名单你的前端域名
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:5173",              # 本地开发
    "https://digit-jet.vercel.app"        # 你的 Vercel 前端域名（替换成实际的）
]}}, supports_credentials=False)

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

# 支持预检：POST + OPTIONS
@app.route("/upload", methods=["POST", "OPTIONS"])
def upload():
    # 预检请求（浏览器为了跨域先发 OPTIONS），直接放行
    if request.method == "OPTIONS":
        return ("", 204)

    t0 = time.time()

    # 取 JSON，容错一些
    data = request.get_json(silent=True) or {}
    pixels = data.get("pixels")

    if not isinstance(pixels, list) or len(pixels) != 28 * 28:
        return jsonify({"ok": False, "error": "Expected 'pixels' length 784"}), 400

    X = preprocess_pixels(pixels)
    pred, probs = net.predict(X)

    return jsonify({
        "ok": True,
        "pred": int(pred[0]),
        "probs": probs[0].tolist(),
        "time_ms": int((time.time() - t0) * 1000)
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render 会注入 PORT
    app.run(host="0.0.0.0", port=port, debug=False)
