# In[1]:
import numpy as np
import perfplot
from functools import reduce


def compare(n):
    compareL = np.full(np.shape(n), np.inf)
    for i in range(4):
        for index in range(len(n)):
            if n[index] < compareL[index]:
                compareL[index] = n[index]


def compare_num(n):
    reduce(np.minimum, [n, n, n, n])


b = perfplot.bench(
    equality_check=None,
    setup=lambda n: np.random.rand(n),
    kernels=[compare_num, compare],
    labels=["numpy.minimum() + reduce", "if condition + for loop"],
    n_range=[2 ** k for k in range(50)],
    xlabel="len(a)",
)
b.show()

# %%
