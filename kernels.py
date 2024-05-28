import abc

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance


class Kernel(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: NDArray, y: NDArray) -> float:
        ...


class Linear(Kernel):
    def __call__(self, x: NDArray, y: NDArray) -> float:
        return x @ y


class RBF(Kernel):
    def __init__(self, *, gamma = 0.1) -> None:
        super().__init__()

        self.gamma = gamma

    def __call__(self, x: NDArray, y: NDArray) -> float:
        norma = (x - y) @ (x - y)

        return np.exp(-(norma / (2 * self.gamma**2)))
