
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial


class here:

    @staticmethod
    def summer(num, arr):
        print(num)
        return sum(arr)

arrs = np.array([np.array([1,2,3]),np.array([3,4,5]),np.array([6,7,8])])

with Pool(cpu_count()) as pool:
    const = 0
    funct = partial(here.summer, const)
    sums = pool.map(funct, arrs)

    pool.close()
    pool.join()

print(sums)