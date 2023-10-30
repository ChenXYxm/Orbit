import numpy as np
a1 = np.array([17,9])
a2 = np.array([17,0])
a3 = np.array([0,0])
a4 = np.array([0,9])
l1 = a2 - a1
l2 = a4 - a1
print(np.cross(l2,l1),np.sign(np.cross(l1,l2)))
print(np.arctan2(-.5,.5),np.arctan2(.5,-.5),np.arctan2(-.5,-.5),np.arctan2(.5,.5))