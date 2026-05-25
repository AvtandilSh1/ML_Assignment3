from builtins import range
import numpy as np


# აფინური შრე

def affine_forward(x, w, b):
    N = x.shape[0]
    x_flat = x.reshape(N, -1)
    out = x_flat.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    x, w, b = cache
    N = x.shape[0]
    x_flat = x.reshape(N, -1)
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x_flat.T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


# ReLU აქტივაცია

def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    dx = dout * (x > 0)
    return dx


# Softmax დანაკარგი

def softmax_loss(x, y):
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


# ბეჩ ნორმალიზაცია

def batchnorm_forward(x, gamma, beta, bn_param):
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)
    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))
    out, cache = None, None

    if mode == "train":
        mu = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        x_norm = (x - mu) / np.sqrt(var + eps)
        out = gamma * x_norm + beta
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var
        cache = (x, x_norm, mu, var, gamma, beta, eps)
    elif mode == "test":
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var
    return out, cache


def batchnorm_backward(dout, cache):
    x, x_norm, mu, var, gamma, beta, eps = cache
    N, D = x.shape
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    dx_norm = dout * gamma
    dvar = np.sum(dx_norm * (x - mu) * -0.5 * (var + eps) ** (-1.5), axis=0)
    dmu = np.sum(dx_norm * -1 / np.sqrt(var + eps), axis=0) + dvar * np.sum(-2 * (x - mu), axis=0) / N
    dx = dx_norm / np.sqrt(var + eps) + dvar * 2 * (x - mu) / N + dmu / N
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    x, x_norm, mu, var, gamma, beta, eps = cache
    N = x.shape[0]
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    dx_norm = dout * gamma
    dx = (1.0 / N) / np.sqrt(var + eps) * (N * dx_norm - np.sum(dx_norm, axis=0) - x_norm * np.sum(dx_norm * x_norm, axis=0))
    return dx, dgamma, dbeta


# ლეიერ ნორმალიზაცია

def layernorm_forward(x, gamma, beta, ln_param):
    eps = ln_param.get("eps", 1e-5)
    mu = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    x_norm = (x - mu) / np.sqrt(var + eps)
    out = gamma * x_norm + beta
    cache = (x, x_norm, mu, var, gamma, beta, eps)
    return out, cache


def layernorm_backward(dout, cache):
    x, x_norm, mu, var, gamma, beta, eps = cache
    N, D = x.shape
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)
    dx_norm = dout * gamma
    dx = (1.0 / D) / np.sqrt(var + eps) * (D * dx_norm - np.sum(dx_norm, axis=1, keepdims=True) - x_norm * np.sum(dx_norm * x_norm, axis=1, keepdims=True))
    return dx, dgamma, dbeta


# დროპაუტი

def dropout_forward(x, dropout_param):
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])
    mask = None
    out = None

    if mode == "train":
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
    elif mode == "test":
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache


def dropout_backward(dout, cache):
    dropout_param, mask = cache
    mode = dropout_param["mode"]
    dx = None
    if mode == "train":
        dx = dout * mask
    elif mode == "test":
        dx = dout
    return dx


# კონვოლუციური შრე

def conv_forward_naive(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    out = np.zeros((N, F, H_out, W_out))

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    out[n, f, i, j] = np.sum(
                        x_padded[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] * w[f]
                    ) + b[f]

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    dx_padded = np.zeros_like(x_padded)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    db[f] += dout[n, f, i, j]
                    dw[f] += x_padded[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] * dout[n, f, i, j]
                    dx_padded[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += w[f] * dout[n, f, i, j]

    dx = dx_padded[:, :, pad:pad+H, pad:pad+W]
    return dx, dw, db


# მაქს პულინგი

def max_pool_forward_naive(x, pool_param):
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    out = np.zeros((N, C, H_out, W_out))
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    out[n, c, i, j] = np.max(
                        x[n, c, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
                    )

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    dx = np.zeros_like(x)
    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    region = x[n, c, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
                    mask = (region == np.max(region))
                    dx[n, c, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width] += dout[n, c, i, j] * mask

    return dx


# სივრცული ბეჩ ნორმალიზაცია

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    N, C, H, W = x.shape
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)
    out_reshaped, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    out = out_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    N, C, H, W = dout.shape
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx_reshaped, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
    dx = dx_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return dx, dgamma, dbeta


# სივრცული ჯგუფური ნორმალიზაცია

def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    eps = gn_param.get("eps", 1e-5)
    N, C, H, W = x.shape
    x_reshaped = x.reshape(N, G, C // G, H, W)
    mu = np.mean(x_reshaped, axis=(2, 3, 4), keepdims=True)
    var = np.var(x_reshaped, axis=(2, 3, 4), keepdims=True)
    x_norm = (x_reshaped - mu) / np.sqrt(var + eps)
    x_norm = x_norm.reshape(N, C, H, W)
    out = gamma * x_norm + beta
    cache = (x, x_norm, mu, var, gamma, beta, G, eps)
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    x, x_norm, mu, var, gamma, beta, G, eps = cache
    N, C, H, W = dout.shape

    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    dgamma = np.sum(dout * x_norm, axis=(0, 2, 3), keepdims=True)

    dx_norm = dout * gamma
    dx_norm_reshaped = dx_norm.reshape(N, G, C // G, H, W)
    x_reshaped = x.reshape(N, G, C // G, H, W)
    M = (C // G) * H * W

    dvar = np.sum(dx_norm_reshaped * (x_reshaped - mu) * -0.5 * (var + eps) ** (-1.5), axis=(2, 3, 4), keepdims=True)
    dmu = np.sum(dx_norm_reshaped * -1 / np.sqrt(var + eps), axis=(2, 3, 4), keepdims=True)
    dmu += dvar * np.sum(-2 * (x_reshaped - mu), axis=(2, 3, 4), keepdims=True) / M

    dx = dx_norm_reshaped / np.sqrt(var + eps) + dvar * 2 * (x_reshaped - mu) / M + dmu / M
    dx = dx.reshape(N, C, H, W)

    return dx, dgamma, dbeta
