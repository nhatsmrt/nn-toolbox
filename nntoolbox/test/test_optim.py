import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import Callable


__all__ = ['test_rosenbrock']


def rosenbrock(x) -> Tensor:
    return (1.0 - x[:-1]).pow(2).sum() + 100 * (x[1:] - x[:-1].pow(2)).pow(2).sum()


def test_rosenbrock(
        optimizer: Callable[..., Optimizer], dim: int=3, eps: float=1e-6, verbose: bool=True,
        max_iter: int=1e10
) -> int:
    """
    Example use:

    >>> test_rosenbrock(torch.optim.Adam)

    :param optimizer: A function that returns an optimizer given a list of parameters/variables
    :param dim: dimension of Rosenbrock function
    :param eps: minimum value of loss to complete training
    :param verbose: whether to print current function and input value every iteration
    :param max_iter: maximum number of iteration to test
    :return: number of iteration to bring function value below eps
    """
    variable = torch.rand(dim, requires_grad=True)
    optimizer = optimizer([variable])
    iter = 0
    fn = float('inf')

    while fn > eps:
        optimizer.zero_grad()
        fn = rosenbrock(variable)
        fn.backward()
        optimizer.step()
        iter += 1
        fn = fn.detach().numpy()

        if verbose:
            print("Current value: " + str(fn))
            print("Input: ")
            print(variable)
            print()

        if iter > max_iter:
            print("Maximum number of iteration reached without converging to the minimum.")
            return -1
    print("Optimizer takes " + str(iter) + " iterations to complete Rosenbrock test.")
    return iter
