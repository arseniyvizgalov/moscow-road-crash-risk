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

def numerical_gradient(X, y, w, b, eps=1e-6):
    """Estimate the gradient without calculus, to check gradients() above.
    Nudge one weight up and down by eps and see how the loss changes:
        dL/dw_j ≈ (L(w_j + eps) - L(w_j - eps)) / (2·eps)
    Slow (2 forward passes per weight), so use it only on a small sample."""
    num_dw = np.zeros_like(w)
    for j in range(len(w)):
        w_plus, w_minus = w.copy(), w.copy()
        w_plus[j] += eps
        w_minus[j] -= eps
        num_dw[j] = (loss(predict_proba(X, w_plus, b), y) - loss(predict_proba(X, w_minus, b), y)) / (2 * eps)
    num_db = (loss(predict_proba(X, w, b + eps), y) - loss(predict_proba(X, w, b - eps), y)) / (2 * eps)
    return num_dw, num_db
