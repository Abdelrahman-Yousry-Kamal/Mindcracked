from torch import nn
from torch.nn.utils.parametrize import register_parametrization

from files.layers.parametrization import MaxNorm, MaxNormParametrize


class MaxNormLinear(nn.Linear):
    """Linear layer with MaxNorm constraining on weights.
    keras version -> EEGNet
    """

    def __init__(
        self, in_features, out_features, bias=True, max_norm_val=2, eps=1e-5, **kwargs
    ):
        super().__init__(
            in_features=in_features, out_features=out_features, bias=bias, **kwargs
        )
        self._max_norm_val = max_norm_val
        self._eps = eps
        register_parametrization(self, "weight", MaxNorm(self._max_norm_val, self._eps))


class LinearWithConstraint(nn.Linear):
    """Linear layer with max-norm constraint on the weights.
    """

    def __init__(self, *args, max_norm=1.0, **kwargs):
        super(LinearWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm
        register_parametrization(self, "weight", MaxNormParametrize(self.max_norm))


class Conv2dWithConstraint(nn.Conv2d):
    """2D convolution with max-norm constraint on the weights.
    """

    def __init__(self, *args, max_norm=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm
        # initialize the weights
        nn.init.xavier_uniform_(self.weight, gain=1)
        register_parametrization(self, "weight", MaxNormParametrize(self.max_norm))
