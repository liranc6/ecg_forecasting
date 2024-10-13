import torch
import torch.nn as nn

class RevIN(nn.Module):
    """
    RevIN performs reversible normalization and denormalization of input data.
    It can learn affine transformations and is designed to stabilize training by normalizing input features.

    Attributes:
        args: Configuration arguments for the module.
        num_features: The number of features or channels in the input.
        eps: A small value added for numerical stability during normalization.
        affine: A boolean indicating if learnable affine parameters should be used.
        subtract_last: A boolean indicating if the last time step should be subtracted during normalization.

    Args:
        args: Configuration arguments for the module.
        num_features (int): The number of features or channels.
        eps (float, optional): A value added for numerical stability. Default is 1e-5.
        affine (bool, optional): If True, RevIN has learnable affine parameters. Default is True.
        subtract_last (bool, optional): If True, the last time step is subtracted during normalization. Default is False.
    """
    def __init__(self, args, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.args = args
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'test_norm':
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            if self.args.training.misc.subtract_short_terms:
                assert False, "you shul not pass"
                self.mean = torch.mean(x[:,-self.args.training.sequence.deprecated_label_len:,:], dim=dim2reduce, keepdim=True).detach()  # TODO: is it suposed to be .deprecated_label_len??
            else:
                self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x