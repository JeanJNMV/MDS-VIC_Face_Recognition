from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Model(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class Eigenfaces:
    def __init__(self, K: int):
        self.K = K
        self.pca = PCA(n_components=K, svd_solver="randomized", whiten=False)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        # X: (N, H, W) # noqa : ERA001
        X = X.reshape(X.shape[0], -1)  # (N, D)
        self.pca.fit(X)

        self.mean_face = self.pca.mean_  # (D,)
        self.components = self.pca.components_

        self.A_train = self.pca.transform(X)  # (N, K)
        self.labels = Y

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X: (N , H, W) # noqa : ERA001
        y_pred = []
        for img in X:
            Xc = img.reshape(-1) - self.mean_face
            A_test = self.components @ Xc
            dists = np.linalg.norm(self.A_train - A_test[None, :], axis=1)
            idx = np.argmin(dists)
            y_pred.append(self.labels[idx])

        return np.array(y_pred)


class LBPH:
    def __init__(
        self,
        P: int = 8,
        R: int = 2,
        grid: Tuple[int, int] = (7, 7),
        use_uniform: bool = True,
        weights: Optional[np.ndarray] = None,
    ):
        """Parameters
        ----------
        P : int
            Number of sampling points (default: 8)
        R : int
            Radius (default: 2)
        grid : (int, int)
            Grid division of the image (rows, cols)
        use_uniform : bool
            Use uniform LBP patterns (LBPu2)
        weights : ndarray or None
            Optional region weights (shape = grid)
        eps : float
            Stability constant for chi-square distance

        """
        self.P = P
        self.R = R
        self.grid = grid
        self.use_uniform = use_uniform
        self.weights = weights

        self._uniform_map = self._build_uniform_pattern_map(P) if use_uniform else None
        self._n_bins = P * (P - 1) + 3 if use_uniform else 2**P

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # X : (N, H, W); y : (N,)
        self.labels = y
        self.histograms = np.array([self._extract_features(img) for img in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        # X : (N, H, W)
        y_pred = []

        for img in X:
            hist = self._extract_features(img)
            dists = np.array([self._chi_square(hist, train_hist) for train_hist in self.histograms])
            idx = np.argmin(dists)
            y_pred.append(self.labels[idx])

        return np.array(y_pred)

    def _extract_features(self, img: np.ndarray) -> np.ndarray:
        lbp = self._lbp_image(img)
        return self._spatial_histogram(lbp)

    def _lbp_image(self, img: np.ndarray) -> np.ndarray:
        H, W = img.shape
        lbp = np.zeros((H, W), dtype=np.int32)

        angles = 2 * np.pi * np.arange(self.P) / self.P
        dy = -self.R * np.sin(angles)
        dx = self.R * np.cos(angles)

        for y in range(self.R, H - self.R):
            for x in range(self.R, W - self.R):
                center = img[y, x]
                code = 0

                for i in range(self.P):
                    yy = y + dy[i]
                    xx = x + dx[i]
                    val = self._bilinear_interpolate(img, yy, xx)
                    code |= (val >= center) << i

                if self.use_uniform:
                    lbp[y, x] = self._uniform_map[code]
                else:
                    lbp[y, x] = code

        return lbp

    def _spatial_histogram(self, lbp: np.ndarray) -> np.ndarray:
        H, W = lbp.shape
        gh, gw = self.grid
        h_step = H // gh
        w_step = W // gw

        histograms = []

        for i in range(gh):
            for j in range(gw):
                region = lbp[i * h_step : (i + 1) * h_step, j * w_step : (j + 1) * w_step]
                hist, _ = np.histogram(
                    region,
                    bins=self._n_bins,
                    range=(0, self._n_bins),
                )
                histograms.append(hist)

        histograms = np.array(histograms).astype(np.float32)

        # Normalize each region's histogram
        for i in range(len(histograms)):
            hist_sum = np.sum(histograms[i])
            if hist_sum > 0:
                histograms[i] /= hist_sum

        if self.weights is not None:
            histograms = histograms * self.weights.reshape(-1, 1)

        return histograms.flatten()

    def _chi_square(self, h1: np.ndarray, h2: np.ndarray) -> float:
        return np.sum((h1 - h2) ** 2 / (h1 + h2 + 1e-10))

    @staticmethod
    def _bilinear_interpolate(img, y, x):
        y0, x0 = int(np.floor(y)), int(np.floor(x))
        y1, x1 = min(y0 + 1, img.shape[0] - 1), min(x0 + 1, img.shape[1] - 1)

        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)

        return wa * img[y0, x0] + wb * img[y0, x1] + wc * img[y1, x0] + wd * img[y1, x1]

    @staticmethod
    def _build_uniform_pattern_map(P: int) -> np.ndarray:
        mapping = np.zeros(2**P, dtype=np.int32)
        uniform_count = 0

        # First pass: count and assign uniform patterns
        for i in range(2**P):
            b = ((i >> np.arange(P)) & 1).astype(np.int32)
            transitions = np.sum(b[:-1] != b[1:]) + (b[-1] != b[0])

            if transitions <= 2:
                mapping[i] = uniform_count
                uniform_count += 1

        # Second pass: assign non-uniform patterns to the last bin
        non_uniform_bin = uniform_count
        for i in range(2**P):
            b = ((i >> np.arange(P)) & 1).astype(np.int32)
            transitions = np.sum(b[:-1] != b[1:]) + (b[-1] != b[0])

            if transitions > 2:
                mapping[i] = non_uniform_bin

        return mapping


class TinyCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 28 * 23, 128)  # ORL size: 112×92 → /4
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DeepLearningModel(Model):
    def __init__(self):
        self.model = TinyCNN(40)

    def fit(self, X: np.ndarray, y: np.ndarray, n_epochs: int = 10, verbose=False) -> None:
        self.model.train()
        y = y - 1
        Xtr_t = torch.from_numpy(X).float().unsqueeze(1) / 255.0
        ytr_t = torch.from_numpy(y).long()

        train_loader = DataLoader(
            TensorDataset(Xtr_t, ytr_t),
            batch_size=32,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(n_epochs):
            self.model.train()
            running_loss = 0.0

            for x, y in train_loader:
                optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x.size(0)

            train_loss = running_loss / len(train_loader.dataset)
            if verbose:
                print(f"Epoch [{epoch + 1:02d}/{n_epochs}] Train loss: {train_loss:.4f}")

    def predict(self, X: np.ndarray):
        self.model.eval()

        # Preprocess
        X_t = torch.from_numpy(X).float().unsqueeze(1) / 255.0
        preds = []

        with torch.no_grad():
            for i in range(0, len(X_t), 32):
                x = X_t[i : i + 32]
                logits = self.model(x)
                preds.append(logits.argmax(dim=1).cpu())

        return torch.cat(preds).numpy() + 1
