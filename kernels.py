import abc

from numpy.typing import NDArray


class Kernel(abc.ABC):
    @abc.abstractmethod
    def __call__(self, x: NDArray, y: NDArray) -> float:
        ...


class Linear(Kernel):
    def __call__(self, x: NDArray, y: NDArray) -> float:
        return x @ y
