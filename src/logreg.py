import numpy as np


def sigmoid(z):
    """Score -> probability: p = 1 / (1 + e^(-z))."""
    z = np.clip(z, -500, 500)            # e^(-z) overflows for huge z; clip keeps it safe
    return 1.0 / (1.0 + np.exp(-z))


def loss(p, y):
    """Binary cross-entropy: L = -(1/n) Σ [y·ln p + (1-y)·ln(1-p)]."""
    p = np.clip(p, 1e-12, 1 - 1e-12)     # ln(0) = -infinity, so never let p be exactly 0 or 1
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def gradients(X, y, p):
    """dL/dw = (1/n)·Xᵀ(p - y),   dL/db = mean(p - y)."""
    error = p - y                        # prediction minus truth, one number per crash
    dw = X.T @ error / len(y)            # each feature collects the errors of crashes where it was 1
    db = error.mean()
    return dw, db


def predict_proba(X, w, b):
    """Forward pass for all crashes: z = Xw + b, p = σ(z)."""
    return sigmoid(X @ w + b)


def train(X, y, lr=0.5, epochs=1000, X_val=None, y_val=None):
    n, d = X.shape
    w = np.zeros(d)                      # start knowing nothing: all weights 0
    b = 0.0
    history = {"train": [], "val": []}

    for epoch in range(epochs):
        p = predict_proba(X, w, b)                       # 1. forward: predict
        history["train"].append(loss(p, y))              # 2. measure how wrong
        if X_val is not None:
            history["val"].append(loss(predict_proba(X_val, w, b), y_val))

        dw, db = gradients(X, y, p)                      # 3. backward: gradient
        w -= lr * dw                                     # 4. step against the gradient
        b -= lr * db

        if epoch % 100 == 0:
            print(f"epoch {epoch}: train loss {history['train'][-1]:.4f}")

    return w, b, history