# In[1]:
import timeit
import rt2_copy
import rt3_copy

log = open("benchmarkLog.txt", 'w')

for i in range(1, 51):
    if i % 10 == 0:
        print(i)
    w = 40 * i
    h = 30 * i
    log.write(str(rt2_copy.main(w, h)) + "\n")
    log.write(str(rt3_copy.main(w, h)) + "\n")

log.close()

# %%
