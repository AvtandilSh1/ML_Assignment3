from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


# სრულად დაკავშირებული ნეირონული ქსელი

class FullyConnectedNet(object):

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            self.params[f'W{i+1}'] = np.random.randn(dims[i], dims[i+1]) * weight_scale
            self.params[f'b{i+1}'] = np.zeros(dims[i+1])

        if self.normalization in ('batchnorm', 'layernorm'):
            for i in range(self.num_layers - 1):
                self.params[f'gamma{i+1}'] = np.ones(hidden_dims[i])
                self.params[f'beta{i+1}'] = np.zeros(hidden_dims[i])

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

        # წინ გავრცელება
        caches = []
        out = X
        for i in range(self.num_layers - 1):
            W = self.params[f'W{i+1}']
            b = self.params[f'b{i+1}']
            if self.normalization == 'batchnorm':
                gamma = self.params[f'gamma{i+1}']
                beta = self.params[f'beta{i+1}']
                out, cache = affine_bn_relu_forward(out, W, b, gamma, beta, self.bn_params[i])
            elif self.normalization == 'layernorm':
                gamma = self.params[f'gamma{i+1}']
                beta = self.params[f'beta{i+1}']
                out, cache = affine_ln_relu_forward(out, W, b, gamma, beta, self.bn_params[i])
            else:
                out, cache = affine_relu_forward(out, W, b)
            caches.append(cache)
            if self.use_dropout:
                out, dropout_cache = dropout_forward(out, self.dropout_param)
                caches.append(('dropout', dropout_cache))

        W = self.params[f'W{self.num_layers}']
        b = self.params[f'b{self.num_layers}']
        scores, last_cache = affine_forward(out, W, b)
        caches.append(last_cache)

        if mode == "test":
            return scores

        # უკან გავრცელება
        loss, grads = 0.0, {}

        loss, dscores = softmax_loss(scores, y)
        for i in range(self.num_layers):
            W = self.params[f'W{i+1}']
            loss += 0.5 * self.reg * np.sum(W * W)

        dout, dW, db = affine_backward(dscores, caches[-1])
        grads[f'W{self.num_layers}'] = dW + self.reg * self.params[f'W{self.num_layers}']
        grads[f'b{self.num_layers}'] = db

        for i in range(self.num_layers - 2, -1, -1):
            if self.use_dropout:
                dropout_cache = caches[2 * i + 1][1]
                dout = dropout_backward(dout, dropout_cache)
                cache = caches[2 * i]
            else:
                cache = caches[i]

            if self.normalization == 'batchnorm':
                dout, dW, db, dgamma, dbeta = affine_bn_relu_backward(dout, cache)
                grads[f'gamma{i+1}'] = dgamma
                grads[f'beta{i+1}'] = dbeta
            elif self.normalization == 'layernorm':
                dout, dW, db, dgamma, dbeta = affine_ln_relu_backward(dout, cache)
                grads[f'gamma{i+1}'] = dgamma
                grads[f'beta{i+1}'] = dbeta
            else:
                dout, dW, db = affine_relu_backward(dout, cache)

            grads[f'W{i+1}'] = dW + self.reg * self.params[f'W{i+1}']
            grads[f'b{i+1}'] = db

        return loss, grads
