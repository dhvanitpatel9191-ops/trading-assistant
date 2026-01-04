"""
Python implementation of trading engine functions.
Replaces the C++ engine for compatibility with cloud hosting platforms like Render.
"""
import numpy as np
import math


def calculate_volatility(prices):
    if len(prices) == 0:
        return 0.0

    prices_array = np.asarray(prices, dtype=np.float64).flatten()
    mean = float(np.mean(prices_array))
    variance = float(np.mean((prices_array - mean) ** 2))
    return float(math.sqrt(variance))


def calculate_sma(prices):
    if len(prices) == 0:
        return 0.0

    prices_array = np.asarray(prices, dtype=np.float64).flatten()
    return float(np.mean(prices_array))


def calculate_ema(prices, alpha=0.1):
    if len(prices) == 0:
        return 0.0

    prices_array = np.asarray(prices, dtype=np.float64).flatten()
    ema = float(prices_array[0])
    for i in range(1, len(prices_array)):
        ema = alpha * prices_array[i] + (1 - alpha) * ema
    return float(ema)


def calculate_rsi(prices):
    if len(prices) < 2:
        return 50.0

    prices_array = np.asarray(prices, dtype=np.float64).flatten()
    gain = 0.0
    loss = 0.0
    for i in range(1, len(prices_array)):
        diff = prices_array[i] - prices_array[i - 1]
        if diff > 0:
            gain += diff
        else:
            loss -= diff
    if loss == 0:
        return 100.0
    rs = gain / loss
    return float(100 - (100 / (1 + rs)))


def find_support_resistance(prices):
    if len(prices) < 3:
        return np.array([]), np.array([])

    prices_array = np.asarray(prices, dtype=np.float64).flatten()
    supports = []
    resistances = []

    for i in range(1, len(prices_array) - 1):
        if prices_array[i] < prices_array[i - 1] and prices_array[i] < prices_array[i + 1]:
            supports.append(prices_array[i])
        if prices_array[i] > prices_array[i - 1] and prices_array[i] > prices_array[i + 1]:
            resistances.append(prices_array[i])

    return np.array(supports, dtype=np.float64), np.array(resistances, dtype=np.float64)
