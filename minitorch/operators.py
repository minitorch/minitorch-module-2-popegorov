"""Collection of the core mathematical operators used throughout the code base."""

import math


# ## Task 0.1
from typing import Callable, Iterable, Any

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    return x * y

def id(x: Any) -> Any:
    return x

def add(x: float, y: float) -> float:
    return x + y

def neg(x: float) -> float:
    return -float(x)

def lt(x: float, y: float) -> float:
    return 1.0 if x < y else 0.0

def eq(x: float, y: float) -> float:
    return 1.0 if x == y else 0.0

def max(x: float, y: float) -> float:
    return x if x > y else y

def is_close(x: float, y: float) -> float:
    return abs(x - y) < 1e-2

def exp(x: float) -> float:
    return math.exp(x)

def sigmoid(x: float) -> float:
    return 1.0 / (1 + exp(-x)) if x >= 0 else exp(x)/(1.0 + exp(x))

def relu(x: float) -> float:
    return max(0.0, x)

def log(x: float) -> float:
    if x <= 0:
        print("Log can be computed only with positive argument")
    return math.log(x)

def inv(x: float) -> float:
    if not x:
        print("Devision by zero")
    return 1 / x

def log_back(x: float, c: float) -> float:
    return inv(x) * c

def inv_back(x: float, c: float) -> float:
    return neg(inv(x)**2) * c

def relu_back(x: float, c: float) -> float:
    return c if x >= 0 else 0.0



# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(a: Iterable, fn: Callable) -> Iterable:
    x = a.copy()
    for i, elem in enumerate(a):
        x[i] = fn(elem)
    return x


def zipWith(a: Iterable, b: Iterable) -> Iterable:
    if len(a) != len(b):
        print("Lengths of iterables should be equal")
        return
    x = a.copy()
    for i, elem in enumerate(a):
        x[i] = (elem, b[i])
    return x


def reduce(a: Iterable, fn: Callable, start_value: Any) -> Any:
    result = start_value
    for elem in a:
        result = fn(result, elem)
    return result


def negList(a: Iterable) -> Iterable:
    return map(a, neg)


def addLists(a: Iterable, b: Iterable) -> Iterable:
    z = a.copy()
    for i, (x, y) in enumerate(zipWith(a, b)):
        z[i] = x + y
    return z


def sum(a: Iterable) -> Any:
    return reduce(a, add, 0.0)


def prod(a: Iterable) -> Any:
    return reduce(a, mul, 1.0)


