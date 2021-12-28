# In[1]:
import numpy as np
import perfplot


def compare(n):
    np.where((n > 0) & (n < n), n, np.inf)


def if_cond(n):
    compareL = np.zeros(np.shape(n))
    for index in range(len(n)-1):
        if n[index] > 0:
            compareL[index] = min(n[index], n[index+1])
        else:
            compareL[index] = np.inf


b = perfplot.bench(
    equality_check=None,
    setup=lambda n: np.random.rand(n),
    kernels=[compare, if_cond],
    labels=["numpy.where()", "if condition + min()"],
    n_range=[2 ** k for k in range(50)],
    xlabel="len(a)",
)
b.show()

# %%
