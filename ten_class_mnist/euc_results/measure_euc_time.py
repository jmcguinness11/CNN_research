import os
import time
import numpy as np

'''
#os.system("{ time python load_saved_euc.py ; } > time.txt 2>&1 >/dev/null")
#os.system("cat time.txt | grep elapsed | sed -rn 's/.*m(.*)s.*$/\1 /p'")
#os.system("{ cat time.txt | grep elapsed | sed -rn 's/.*:([0-9]*.[0-9]*)e.*$/\1/p' ; }")
'''

times = []
for k in range(50):
	print("Iteration: " + str(k))
	start = time.time()
	os.system("python load_saved_euc.py >/dev/null 2>1")
	end = time.time()
	print("{0:.3f}".format(end-start))
	times.append(end-start)

times = np.asarray(times)
print("Average time: {0:.3f}".format(np.mean(times)))

#save results
np.save('euc_timing_results.npy', times)
np.savetxt('euc_timing_results.txt', times)
