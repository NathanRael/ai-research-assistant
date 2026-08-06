from typing import Any
import numpy as np


def run_kwarg(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

run_args = lambda *args: [arg for arg in args]

def run_double(data : list[Any]):
    return {index : value*2 for index,value in enumerate(data)}

def test_numpy():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(matrix)

if __name__ == '__main__':
    print(f"Args : {run_args(5,4,3)}")
    print(f"Double : {run_double([5,4,3])}")
    run_kwarg(user='root',password='******', port=5000)
    test_numpy()
