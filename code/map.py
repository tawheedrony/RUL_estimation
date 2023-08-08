from sim import *
from scipy.stats import norm


# MAP to estimate 𝛼, 𝜎 

"""
    𝛼, 𝜎 -> MAP (Maximum A Posteriori Estimation)

    Outputs : 
    𝜃1^ = (𝛼, 𝜎)  
    𝛼 -> System Drift  
    𝜎 -> System Diffusion  

    𝜃1^ = argmax(𝜃1) 𝑝(𝜃1 | 𝛥𝑋) = argmax(𝜃1) 𝑝(𝛥𝑋 | 𝜃1) * 𝑝(𝜃1)    
    Here,  
    𝑝(𝛥𝑋 | 𝜃1) => likelihood of observing 𝛥𝑋 given 𝜃1  
    𝑝(𝜃1) => Prior probability of 𝜃1  

    Steps :
        1. Set prior mean and std of 𝛼 to be 𝛼0 = 9.95, and 𝜎0 = 1.
        2. Set prior mean and std of 𝜎 to be 𝜎𝜇 = 4, and 𝜎1 = 1.
        3. Calculate likelihood and prior probabilities
        4. Calculate MAP
"""
# finding del(X)
n_xc = len(X_c)
del_x = []
for i in range(n_xc-1):
  temp = X_c[i+1] -  X_c[i]
  del_x.append(temp)
del_X = np.array(del_x)


# Prior information
alpha_0 = 9.95
sigma_0 = 1.0
sigma_mu = 4.0
sigma_1 = 1.0


def posteriori_estimate(alpha, sigma):
    likelihood = np.prod(norm.pdf(del_X, loc=alpha*(0.1), scale=sigma*(np.sqrt(0.1))))      # 𝑝(𝛥𝑋 | 𝜃1)
    prior_alpha = norm.pdf(alpha, loc=alpha_0, scale=sigma_0)                               # 𝑝(alpha_0)
    prior_sigma = norm.pdf(sigma, loc=sigma_mu, scale=sigma_1)                              # 𝑝(sigma_mu)
    posteriori_estimate = likelihood * prior_alpha * prior_sigma                            # 𝑝(𝜃1 | 𝛥𝑋) = 𝑝(𝛥𝑋 | 𝜃1) * 𝑝(alpha_0) * 𝑝(sigma_mu)

    return posteriori_estimate

def map_estimation():
    alpha_values = np.linspace(0, 20, 100)                                                  # Define a range of alpha values
    sigma_values = np.linspace(0.1, 10, 100)                                                # Define a range of sigma values

    best_alpha, best_sigma, max_posterior = None, None, -np.inf

    for alpha in alpha_values:
        for sigma in sigma_values:
            p = posteriori_estimate(alpha, sigma)
            if p > max_posterior:
                max_posterior = p
                best_alpha, best_sigma = alpha, sigma

    return best_alpha, best_sigma



if __name__ == "__main__":
    estimated_alpha, estimated_sigma = map_estimation()
    print("Estimated 𝛼:",round(estimated_alpha,2))
    print("Estimated 𝜎:", round(estimated_sigma,2))