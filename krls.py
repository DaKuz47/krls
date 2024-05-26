from dataclasses import dataclass

from typing import Any

import numpy as np
from numpy.typing import NDArray

from kernels import Kernel


@dataclass
class ALDResult:
    is_linear_dep: bool
    a_coeffs: NDArray
    delta: float


class KRLSModel:
    def __init__(
        self,
        c: float,
        v: float,
        kernel: type[Kernel],
        kernel_params: dict[str, Any] | None = None
    ) -> None:
        self.c = c
        self.v = v
        self.kernel = kernel(**(kernel_params or {}))
        self.need_init_data = True
        self.inv_kernel_mtx = None
        self.p_mtx = None
        self.dual_coefs = None
        self.dictionary = []

    def init_data(self, x: NDArray) -> None:
        assert self.need_init_data
        assert x.ndim == 1

        k00 = self.kernel(x, x)
        self.inv_kernel_mtx = np.array([[1 / k00]])
        self.p_mtx = np.array([[1]])
        self.dictionary.append(x)

        self.need_init_data = False

    def clear_data(self) -> None:
        self.p_mtx = None
        self.inv_kernel_mtx = None
        self.dictionary = []

        self.need_init_data = True

    def kernel_row(self, x: NDArray) -> NDArray:
        return np.array([self.kernel(sv, x) for sv in self.dictionary])

    def ald_test(self, x: NDArray) -> ALDResult:
        kernel_row = self.kernel_row(x)

        a_cfs = self.inv_kernel_mtx @ kernel_row
        delta = self.kernel(x, x) - kernel_row @ a_cfs

        return ALDResult(
            is_linear_dep=delta <= self.v,
            a_coeffs=a_cfs,
            delta=delta,
        )

    def fit(self, x: NDArray, y: NDArray):
        assert len(x) > 0, 'Empty data'

        if y.ndim > 1:
            assert len(y) == 1
            y = y.ravel()

        assert len(x) == len(y), 'x and y sizes must be same'

        start_idx = 0
        if self.need_init_data:
            self.init_data(x[0])
            start_idx = 1

        curr_alpha = [y[0] / self.kernel(x[0], x[0])]

        for curr_x, curr_y in zip(x[start_idx:], y[start_idx:]):
            ald_result = self.ald_test(curr_x)
            a_cfs = ald_result.a_coeffs
            kernel_row = self.kernel_row(curr_x)

            if ald_result.is_linear_dep:
                q = (self.p_mtx @ a_cfs) / (1 + a_cfs @ self.p_mtx @ a_cfs)
                self.p_mtx = self.p_mtx - (q @ a_cfs) * self.p_mtx

                curr_alpha = curr_alpha + self.inv_kernel_mtx @ (
                    q * (curr_y - kernel_row @ curr_alpha)
                )
            else:
                top_inv_k_part = np.hstack(
                    (
                        ald_result.delta * self.inv_kernel_mtx + a_cfs @ a_cfs,
                        -a_cfs[:, np.newaxis]
                    )
                )

                bottom_inv_k_part = np.append(-a_cfs, 1)

                self.inv_kernel_mtx = (1 / ald_result.delta) * (
                    np.vstack((top_inv_k_part, bottom_inv_k_part))
                )

                self.p_mtx = np.vstack((
                    np.hstack((self.p_mtx, np.zeros(self.p_mtx.shape[0])[np.newaxis, :])),
                    np.append(np.zeros(self.p_mtx.shape[1]), 1)
                ))

                tmp_var = curr_y - kernel_row @ curr_alpha
                curr_alpha = np.append(
                    curr_alpha - (a_cfs / ald_result.delta) * tmp_var,
                    (1 / ald_result.delta) * tmp_var
                )

                self.dictionary.append(curr_x)

        self.dual_coefs = curr_alpha

    def predict(self, x: NDArray):
        return np.array([
            self.dual_coefs @ self.kernel_row(x_outer)
            for x_outer in x
        ])
