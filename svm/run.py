import numpy as np
import matplotlib.pyplot as plt

xvec = np.linspace(-5.,5.,100)                               
x,y = np.meshgrid(xvec, xvec)
z = -np.hypot(x, y)                                

plt.contourf(x, y, z, 100)                             
plt.colorbar() 

plt.axhline(0, color='white')
plt.axvline(0, color='white')

plt.show()