from sim import *
import pylab

set_implementation = True # Set True for KF - 1D, False for KF - 2D

estimated_alpha = 9.9
estimated_sigma = 4.9
estimated_beta = 4.59

#@raisa
class KalmanFilter(object):
    def __init__(self, A = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(A is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = A.shape[1]
        self.m = H.shape[1]

        self.A = A
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self):
        self.x = np.dot(self.A, self.x) + self.B
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

def KF2D():
  dt = 0.0001
  A = np.array([[1, 0],[0,1]])
  B = np.array([[estimated_alpha*dt],[estimated_beta*dt]])
  H = np.array([1, 1]).reshape(1, 2)
  Q = np.array([[(estimated_sigma**2)*dt, 0],[0,(eta**2)*dt]])
  R = np.array([0.46]).reshape(1, 1)
  P = np.array([[0,0],[0,0]])

  measurements = Y

  kf = KalmanFilter(A = A, B=B, H = H, Q = Q, R = R, P=P)
  predictions = []

  for z in measurements:
    predictions.append(np.dot(H,  kf.predict())[0])
    kf.update(z)

  plt.plot(range(len(measurements)), measurements, color = 'r', label = 'Actual System Degradation Simulated Data')
  plt.plot(range(len(predictions)), np.array(predictions), color = 'b', label = 'Estimated System State Using KF')
  plt.legend()
  plt.savefig(os.path.join(out_dir, 'KF2D.png'))
  plt.close()



#@tawheed
def time_update(estimate, error):
    """As there is no control parameter this equation is simplified"""
    return estimate[-1], error[-1]

def measurement_update(measurement, prev_est, prev_error, error_m):
    """Use the estimation to calculate the noise of the data"""
    prev_error = prev_error 
    gain = prev_error / (prev_error + error_m)
    new_est = prev_est + (gain * (measurement - prev_est))
    new_error = (1 - gain) * prev_error
    return new_est, new_error

def KF1D():
    measurements = Y
    error_m=0.1
    estimate=[0]
    error_e=[1]
    for val in measurements:
        prev_est,prev_error=time_update(estimate,error_e)
        new_est,new_error=measurement_update(val,prev_est,prev_error,error_m)
        estimate.append(new_est)
        error_e.append(new_error)
    
    plt.plot(range(len(measurements)), measurements, color = 'r', label = 'Actual System Degradation Simulated Data')
    plt.plot(range(len(estimate)), np.array(estimate), color = 'b', label = 'Estimated System State Using KF')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'KF1D.png'))
    plt.close()


if __name__ == '__main__':
    if set_implementation:
        KF1D()
    else:
        KF2D()