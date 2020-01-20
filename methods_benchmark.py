"""
Created at 20.01.2020

@author: Piotr Bartman
"""

from methods.finite_difference import scheme1, scheme2, scheme3
import math
import time


if __name__ == '__main__':
    schemes = (lambda alpha, dx1, dx2: scheme1(dx=dx1, dy=dx2), scheme2, scheme3)
    alphas = ((None,), (0, .5, 1), (0, .25, .5))
    x1 = ("x", "x", "x")
    dx1s = ((0.25, .1, .01, .001), (0.25, .1, .01, .001), (0.25, .1, .01, .001))
    x2 = ("y", "t", "t")
    dx2s = ((0.25, .1, .01, .001), (.1, .01, .001, .0001), (0.1, 0.01, 0.005, 0.0005))

    for i, scheme in enumerate(schemes):
        for alpha in alphas[i]:
            for dx1 in dx1s[i]:
                for dx2 in dx2s[i]:
                    print(f"scheme{i}[{alpha}](d{x1[i]} = {dx1}, d{x2[i]} = {dx2}): ", end="")
                    mse = math.nan
                    s = math.nan
                    try:
                        s = time.time()
                        mse = scheme(alpha, dx1, dx2)
                        s = time.time() - s
                    except Exception:
                        pass
                    print(f"mse = {mse},time = {round(s, 5)}s" if math.isfinite(mse) else "fail")


