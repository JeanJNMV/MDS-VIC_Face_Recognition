from abc import ABC, abstractmethod

import numpy as np
from sklearn.decomposition import PCA


class Model(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class EigenFaces:
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
