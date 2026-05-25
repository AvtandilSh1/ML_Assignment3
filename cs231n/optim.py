import numpy as np


# ოპტიმიზატორები

def sgd(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    v = config["momentum"] * v - config["learning_rate"] * dw
    next_w = w + v

    config["velocity"] = v
    return next_w, config


def rmsprop(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    config["cache"] = config["decay_rate"] * config["cache"] + (1 - config["decay_rate"]) * dw ** 2
    next_w = w - config["learning_rate"] * dw / (np.sqrt(config["cache"]) + config["epsilon"])

    return next_w, config


def adam(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    config["t"] += 1
    config["m"] = config["beta1"] * config["m"] + (1 - config["beta1"]) * dw
    config["v"] = config["beta2"] * config["v"] + (1 - config["beta2"]) * dw ** 2
    m_hat = config["m"] / (1 - config["beta1"] ** config["t"])
    v_hat = config["v"] / (1 - config["beta2"] ** config["t"])
    next_w = w - config["learning_rate"] * m_hat / (np.sqrt(v_hat) + config["epsilon"])

    return next_w, config
