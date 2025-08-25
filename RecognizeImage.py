import numpy as np
from PIL import Image
from pathlib import Path
import json
import os

# ---------- Image utils ----------
def loadImage(path: Path) -> np.ndarray:

    img = Image.open(path).convert("RGB")
    array = np.asarray(img, dtype=np.float32) / 255.0
    return array

def extract_features(img_rgb: np.ndarray) -> np.ndarray:
    r_mean = img_rgb[..., 0].mean()
    g_mean = img_rgb[..., 1].mean()
    b_mean = img_rgb[..., 2].mean()
    return np.array([r_mean, g_mean, b_mean], dtype=np.float32)  # keep it simple

# ---------- Model (logistic regression scratch) ----------
class LogisticRegressionScratch:
    def __init__(self, n_features: int, lr: float = 0.1):
        self.W = np.random.randn(n_features).astype(np.float32) * 0.01
        self.b = 0.0
        self.lr = lr

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        return self.sigmoid(X @ self.W + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def fit(self, X, y, epochs=500):
        n = X.shape[0]
        for epoch in range(epochs):
            p = self.predict_proba(X)
            grad_w = (X.T @ (p - y)) / n
            grad_b = np.mean(p - y)
            self.W -= self.lr * grad_w
            self.b -= self.lr * grad_b
            if epoch % 100 == 0:
                loss = -np.mean(y*np.log(p+1e-9) + (1-y)*np.log(1-p+1e-9))
                print(f"Epoch {epoch}, Loss={loss:.4f}")

# ---------- Training ----------
def buildDataset(PositiveCase, WorstCase):
    X, y = [], []

    for file in Path(PositiveCase).glob("*.*") :

        img = loadImage(file)
        X.append(extract_features(img))
        y.append(1)

    for file in Path(WorstCase).glob("*.*"):
        img = loadImage(file)
        X.append(extract_features(img))
        y.append(0)

    return np.array(X), np.array(y)


if __name__ == "__main__":

    PositiveCase = r"C:\Users\ajayp\Downloads\Red"
    WorstCase = r"C:\Users\ajayp\Downloads\Not Red"
    TestCase = r"C:\Users\ajayp\Downloads\Test"

    # Train
    X, y = buildDataset(PositiveCase, WorstCase)
    model = LogisticRegressionScratch(n_features=X.shape[1], lr=0.5)
    model.fit(X, y, epochs=500)

    # Test
    print("\n--- Testing on Test folder ---")
    for p in Path(TestCase).glob("*.*"):
        img = loadImage(p)
        feats = extract_features(img)
        pred = model.predict(feats[None, :])[0]
        print(f"{p.name}: {'RED present' if pred==1 else 'NOT RED'}")
