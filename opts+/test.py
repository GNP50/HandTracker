import scan
import numpy as np


a = np.array([[1,2,3],[0,0,6]],dtype=np.uint8)
b = np.array([[0,0,0]],dtype=np.uint8)
print(scan.filterWithArray(a,b,2))



#rand_arr = np.random.randint(low=0, high=0xFF, size=(9, 9), dtype=np.uint8)
#print(f'{rand_arr}')
#print(scan.remove_noise_gs(rand_arr,2,2))



a = np.array([
    [0.2341, 0.7562],
    [0.1233, 0.1344],
    [0.426, 0.738],
    [0.4263245, 0.735434],
    [0.4245245, 0.735432],
    [0.4234252345, 0.732345],
    [0.224, 0.758],
    [0.223, 0.224],
    [0.225, 0.226]
])

print(scan.prepareFourierGrid(a,2));