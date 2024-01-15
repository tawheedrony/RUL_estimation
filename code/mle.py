import pandas as pd
from sim import *
from sympy import symbols, Eq, solve


"""
Measurement increments 𝛥𝑌 follows a multi-variate Gaussian distribution, i.e., 𝛥𝑌 ∼ 𝑁(𝜔𝛥𝑡, 𝛺), 
    where 𝜔 = 𝛼 + 𝛽 and 𝛺 are the variance–covariance matrices.

𝛺 = (𝜎2 + 𝜂2)𝛥𝑡𝑗 + 2𝜎𝜖2 
From 𝛺, we need to find the estimates of 𝜂 and 𝜎. 
But to solve the problem of ‘‘identifiability’’ is to estimate the parameters (𝜂 and 𝜎) with measurements sampled at a different interval.

"""
estimated_alpha = 9.9
estimated_sigma = 4.9

# Monitoring data (X(t)) between the 10th and 11th calibration points (X_c)
lower_lim = np.where(X==X_c[9])[0]
upper_lim = np.where(X==X_c[10])[0]
Y_ = Y[lower_lim[0]:upper_lim[0]]

# del(Y)
n_yc = len(Y_)
del_y = []
for i in range(n_yc-1):
  temp = Y_[i+1] -  Y_[i]
  del_y.append(temp)
del_Y = np.array(del_y)

# Calculating w
J = len(del_Y)
del_t = dt*np.ones(shape = (J,))
w_list_ = []
w = 0
for i in range(J):
  w = w + del_Y[i]/(J * del_t[i])
  w_list_.append(w)

w_list = np.array(w_list_)
print("Estimated 𝜔:",round(w,2))

# plot w

points = np.arange(0, J, 1, dtype=int)
plt.plot(points, w_list, color='b', label='Estimated w')
plt.plot(points, 15*np.ones(shape=(J,)),color= 'r', label = 'Actual w')
plt.xlabel("Monitoring Points")
plt.ylabel("Estimated values of alpha + beta")
plt.title("Estimated w vs the number of data points")
plt.grid()
plt.savefig(os.path.join(out_dir,'w.png' ))


# calculate beta
estimated_beta = w - estimated_alpha
print("Estimated 𝛽:",round(estimated_beta,2))

# calculate varience-covarience
omega_ = (1/J) * ((del_Y - w*del_t).T  @ (del_Y - w*del_t))

# creat omega matrix
diag_arr = omega_*np.ones(shape = (J,))
omega_matrix = np.diag(diag_arr)
omega_det = np.linalg.det(omega_matrix)


#taking measurements sampled at a different interval to calculate eta and sigma_e
del_Y_2 = del_Y[::2]
J_2 = len(del_Y_2)
dt_2 = 2*dt
del_t_2 = dt_2*np.ones(shape = (J_2,))
w_list_2 = []
w_2 = 0
for i in range(J_2):
  w_2 = w_2 + del_Y_2[i]/(J_2 * del_t_2[i])
  w_list_2.append(w_2)

w_list_2 = np.array(w_list_2)
omega_2 = (1/J_2) * ((del_Y_2 - w_2*del_t_2).T  @ (del_Y_2 - w_2*del_t_2))

# create symbols
estimated_eta, estimated_delta_e = symbols('estimated_eta estimated_delta_e')

# defining equation
eq1 = Eq((estimated_sigma**2 + estimated_eta**2)*dt + 2*(estimated_delta_e**2) - omega_ , 0)
eq2 = Eq((estimated_sigma**2 + estimated_eta**2)*dt_2 + 2*(estimated_delta_e**2) - omega_2 , 0)
solution = solve((eq1,eq2), (estimated_eta, estimated_delta_e))

print("Estimated 𝜂:",round(solution[3][0],2))
print("Estimated 𝜎𝜖:",round(solution[3][1],2))