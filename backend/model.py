# backend/model.py
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

class TinyNN:
    def __init__(self, input_dim=784, hidden_dim=64, output_dim=10):
        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, 0.1, (input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros((hidden_dim,), dtype=np.float32)
        self.W2 = rng.normal(0, 0.1, (hidden_dim, output_dim)).astype(np.float32)
        self.b2 = np.zeros((output_dim,), dtype=np.float32)

    def load(self, path="model.npz"):
        data = np.load(path)
        self.W1, self.b1, self.W2, self.b2 = data["W1"], data["b1"], data["W2"], data["b2"]

    def save(self, path="model.npz"):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def forward(self, X):
        h = sigmoid(X @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        probs = softmax(logits)
        return probs

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1), probs

    def train(self, X, y, lr=0.1, batch_size=128, epochs=3):
        Y = np.eye(10, dtype=np.float32)[y]
        n = X.shape[0]
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            Xs, Ys = X[idx], Y[idx]
            for i in range(0, n, batch_size):
                xb = Xs[i:i+batch_size]
                yb = Ys[i:i+batch_size]
                # forward
                h = sigmoid(xb @ self.W1 + self.b1)
                logits = h @ self.W2 + self.b2
                probs = softmax(logits)
                # grad
                dlogits = (probs - yb) / xb.shape[0]
                dW2 = h.T @ dlogits
                db2 = dlogits.sum(axis=0)
                dh = dlogits @ self.W2.T
                dh_raw = dh * h * (1 - h)
                dW1 = xb.T @ dh_raw
                db1 = dh_raw.sum(axis=0)
                # update
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                
            # 小验证（可忽略）
            pred, _ = self.predict(X[:5000])
            acc = (pred == y[:5000]).mean()
            print(f"epoch {epoch+1}/{epochs} acc@5k={acc:.3f}")
