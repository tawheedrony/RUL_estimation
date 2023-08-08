import pandas as pd
import numpy as np
import math
import os 
import matplotlib.pyplot as plt
np.random.seed(40)

"""
    Data Simulation for the paper "Remaining lifetime of degrading systems continuously monitored by degrading sensors" by K. Mukhopadhyay

    Wiener process has been used to describe the degradation processes of the system and the sensor.
    The sum of the two Wiener processes along with measurement error leads to the resultant observations 𝑌(𝑡), 
    which is a deviation from the actual underlying system degradation 𝑋(𝑡)

"""
# set directories
out_dir = 'outputs/'
data_dir = 'data/'


# Parameters

alpha = 10          # system drift 
delta = 5           # system diffusion
beta = 5            #sensor drift
eta = 3             #sensor diffusion
delta_e = 0.45      #measurement error std


# Generate Wiener/Brownian Motion for System and Sensor

n = 21000           # no of points
T = 2.1             # finale time
d = 2               # two independent motions
times = np.linspace(0., T, n)
dt = times[1] - times[0]   
dB = np.sqrt(dt) * np.random.normal(size=(n - 1, d))     
B0 = np.zeros(shape=(1,d))
B = np.concatenate((B0, np.cumsum(dB, axis=0)), axis=0)
plt.plot(times,B)
plt.title('Wiener Process')
plt.legend(['Time(t)','Standard Brownian Motion (B(t))'])
plt.savefig(os.path.join(out_dir,'brownian.png' ))
plt.close()

# Generate System and Sensor Degradation
"""
    System degradation equation: 𝑋(𝑡) = 𝛼𝑡 + 𝜎𝐵1(𝑡)
    Sensor degradation equation: 𝑆(𝑡) = 𝛽𝑡 + 𝜂𝐵2(𝑡)
    Resultant degradation : 𝑌 (𝑡) = 𝑋(𝑡) + 𝑆(𝑡) + 𝜖
"""

times_ = np.reshape(times,(-1,1))
X = alpha*times + delta*B[:,0]
S = beta*times + eta*B[:,1]
Y = X + S + np.random.normal(loc = 0.0, scale = delta_e ,size=(n, ))

# save data
df = pd.DataFrame({'X':X})
df['S'] = pd.Series(S)
df['Y'] = pd.Series(Y)
df.to_csv("data/degradation_data.csv")

# plotting the functions
plt.plot(times, X, color='b', label='System')
plt.plot(times, S, color='g', label='Sensor')
plt.plot(times, Y, color='r', label='Resultant Obsevation')
plt.xlabel("Time")
plt.ylabel("Degradation Level")
plt.title("Simulated Obsevation Data")
plt.legend(['System Degradation','Sensor Degradation','Resultant Obsevation'])
plt.savefig(os.path.join(out_dir,'simulation.png' ))
plt.close()

# calibration sensor data
step = 1000
X_c_ = []
for i in range(0, n, step):
  X_ = X[i]
  X_c_.append(X_)

X_c = np.array(X_c_)

# plot calibration data

t = np.linspace(0., 2.1, 21)
plt.plot(t, X_c, color='b', label='Calibration', marker='*')
plt.xlabel("Time")
plt.ylabel("Degradation Level")
plt.title("Simulated Obsevation Data")
plt.legend(['Calibration Data'])
plt.savefig(os.path.join(out_dir,'calibration.png' ))
plt.close()


if __name__ == "__main__":
    pass