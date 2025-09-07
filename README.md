# 手写数字识别 · （React + Flask + Numpy）

演示 **前端 → 后端 → 模型推理** 的完整链路：前端在浏览器里手写/上传数字，缩成 **28×28 灰度（784 个数）**；后端用 **Numpy 写的 MLP(784→64→10)** 加载 `model.npz` 做预测并返回结果。

---

## 项目结构
    digit/
    ├─ backend/ # 后端（Flask + Numpy）
    │ ├─ app.py # /hello、/upload 接口与服务启动
    │ ├─ model.py # MLP：784→64→10（sigmoid + softmax）
    │ ├─ train.py # 训练 MNIST，生成 model.npz
    │ ├─ utils.py # 预处理（归一化/去噪/简单居中）
    │ └─ model.npz # 训练好的权重（启动时加载）
    └─ frontend/ # 前端（React + Vite）
    ├─ src/
    │ └─ App.jsx # 画布手写/上传→28×28 灰度→调用后端
    ├─ index.html
    └─ package.json


---

## 运行环境
- **Python 3.10+**
- **Node.js 18+**（含 npm）
- Windows / macOS / Linux

---

## 快速开始

### 1) 后端（Flask）
    cd backend
    
    # 建议使用虚拟环境
    python -m venv .venv
    # Windows PowerShell
    .\.venv\Scripts\Activate.ps1
    # 若被执行策略拦截：Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    # macOS/Linux：
    # source .venv/bin/activate
    
    pip install -r requirements.txt
    
    # 启动后端
    python app.py
    # 访问 http://127.0.0.1:5000/hello 应返回 {"ok": true, "msg": "..."}


### 2) 前端（React + Vite）
    cd frontend
    npm install
    npm install axios
    npm run dev
    # 打开终端打印的地址（通常 http://localhost:5173）

---

## 工作流程
    1.前端 Canvas 黑底白字；用户手写或上传图片。
    
    2.将画布压缩为 28×28 灰度（得到 784 个数）。
    
    3.以 JSON POST /upload 到后端。
    
    4.后端做归一化/去噪（utils.py），加载 model.npz 的权重（model.py 的 MLP）前向计算。
    
    5.返回预测与概率；前端展示，并把结果写入 localStorage 作为“历史”（仅本机，不上传）。