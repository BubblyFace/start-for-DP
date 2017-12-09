import numpy as np


test = np.array([1,2,3,4,5])
test = test - np.max(test, axis = 1).reshape(-1,1)

print(test)