import torch
from jaxtyping import Float32
from torch import Tensor


class RunningStandardization:

    def __init__(
        self: "RunningStandardization",
        x_shape: tuple[int, ...],
    ) -> None:
        self.mean: Float32[Tensor, " *x_shape"] = torch.zeros(size=x_shape)
        self.var: Float32[Tensor, " *x_shape"] = torch.zeros(size=x_shape)
        self.std: Float32[Tensor, " *x_shape"] = torch.zeros(size=x_shape)
        self.n: Float32[Tensor, " 1"] = torch.zeros(size=(1,))

    def __call__(
        self: "RunningStandardization",
        x: Float32[Tensor, " *x_shape"],
    ) -> Float32[Tensor, " *x_shape"]:
        self.n += torch.ones(size=(1,))
        new_mean: Float32[Tensor, " *x_shape"] = (
            self.mean + (x - self.mean) / self.n
        )
        new_var: Float32[Tensor, " *x_shape"] = self.var + (x - self.mean) * (
            x - new_mean
        )
        new_std: Float32[Tensor, " *x_shape"] = torch.sqrt(new_var / self.n)
        self.mean, self.var, self.std = new_mean, new_var, new_std
        standardized_x: Float32[Tensor, " *x_shape"] = (x - self.mean) / (
            self.std + self.std.eq(0)
        )
        return standardized_x
