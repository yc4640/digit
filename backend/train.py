# backend/train.py
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from model import TinyNN

def load_mnist():
    print("Downloading MNIST（首次会下，之后走缓存）...")
    X, y = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto',
                        cache=True, return_X_y=True)
    X = (X.astype(np.float32) / 255.0)
    y = y.astype(np.int64)
    return X, y

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.1)
    args = ap.parse_args()

    X, y = load_mnist()
    if args.subset < len(X):
        idx = np.random.default_rng(0).choice(len(X), size=args.subset, replace=False)
        X, y = X[idx], y[idx]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.1, random_state=42)

    net = TinyNN()
    net.train(Xtr, ytr, lr=args.lr, epochs=args.epochs)
    pred, _ = net.predict(Xte)
    acc = (pred == yte).mean()
    print(f"test accuracy ~ {acc:.3f}")
    net.save("model.npz")
    print("Saved weights to model.npz")
