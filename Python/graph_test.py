import matplotlib.pyplot as plt
import numpy as np

x=np.array([1, -1, 2, -2])
y=np.array([1, -1, 2, -2])
fig=plt.figure()
ax=fig.add_subplot(111)
#x,y=[0],[0]
#scatter = ax.scatter([],[],label="SDR #1")
scatter = ax.scatter(x, y)
ax.grid(True)
ax.set_xlabel("Real part")
ax.set_ylabel("Imaginary part")
plt.show()