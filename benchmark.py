# In[1]:
import method3
import method1

log = open("benchmarkLog_4.txt", 'w')

for j in range(1, 31):
    print(j)
    log.write("object " + str(j) + "\n")
    for i in range(1, 51):
        w = 40 * i
        h = 30 * i
        log.write(str(method1.main(w, h, j)) + "\n")
        #log.write(str(rt2_copy.main(w, h, i)) + "\n")
        log.write(str(method3.main(w, h, j)) + "\n")
    log.close()

# %%
