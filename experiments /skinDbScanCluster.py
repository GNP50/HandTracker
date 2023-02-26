import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib



skin = np.load('clusteredHands.npy')

X = np.array(skin[...,0])
Z = np.array(skin[...,1])
Y = np.array(skin[...,2])
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

model = DBSCAN(eps=2,min_samples=2)
a = model.fit(X.reshape(-1,1))
d = model.fit(Y.reshape(-1,1))
c = model.fit(Z.reshape(-1,1))


for i in set(a.labels_):
    cmp = X.reshape(-1,1)[np.where(a.labels_==i)]
    print(f'\"H_max\":{cmp.max()},\"H_min\":{cmp.min()},')


for i in set(d.labels_):
    cmp = Y.reshape(-1,1)[np.where(d.labels_==i)]
    print(f'\"V_max\":{cmp.max()},\"V_min\":{cmp.min()},')

for i in set(c.labels_):
    cmp = Z.reshape(-1,1)[np.where(c.labels_==i)]
    print(f'\"S_max\":{cmp.max()},\"S_Min\":{cmp.min()},')

thr = 20
thrS = 100
skin[..., 0] = X / 180
skin[..., 1] = Y / 255
skin[..., 2] = Z / 255

colors = matplotlib.colors.hsv_to_rgb(skin)

print()
ax.scatter(X,Y,Z,facecolors=colors)
plt.show()

