# RUL Estimation
Remaining lifetime of degrading systems continuously monitored by degrading sensors

## 5.1 Data Simulation for the degradation process
System degradation equation: 𝑋(𝑡) = 𝛼𝑡 + 𝜎𝐵1(𝑡)
Sensor degradation equation: 𝑆(𝑡) = 𝛽𝑡 + 𝜂𝐵2(𝑡)
Resultant degradation : 𝑌 (𝑡) = 𝑋(𝑡) + 𝑆(𝑡) + 𝜖

## 5.2 Parameter Estimation of the degradation Process
**𝛼, 𝜎 -> MAP (Maximum A Posteriori Estimation)**  
**Inputs :** Calibration data (𝛥𝑋/X_c)  

**Outputs :**   
𝜃1^ = (𝛼, 𝜎)  
𝛼 -> System Drift  
𝜎 -> System Diffusion  

𝜃1^ = argmax(𝜃1) 𝑝(𝜃1 | 𝛥𝑋) = argmax(𝜃1) 𝑝(𝛥𝑋 | 𝜃1) * 𝑝(𝜃1)    
Here,  
𝑝(𝛥𝑋 | 𝜃1) => likelihood of observing 𝛥𝑋 given 𝜃1  
𝑝(𝜃1) => Prior probability of 𝜃1  

**Steps :**
1. Set prior mean and std of 𝛼 to be 𝛼0 = 9.95, and 𝜎0 = 1.
2. Set prior mean and std of 𝜎 to be 𝜎𝜇 = 4, and 𝜎1 = 1.
3. Calculate likelihood and prior probabilities
4. Calculate MAP

**𝛽, 𝜂 and 𝜎𝜖 -> MLE (Maximum Likelihood Estimation)**  

Measurement increments 𝛥𝑌 follows a multi-variate Gaussian distribution, i.e., 𝛥𝑌 ∼ 𝑁(𝜔𝛥𝑡, 𝛺), 
    where 𝜔 = 𝛼 + 𝛽 and 𝛺 are the variance–covariance matrices.

𝛺 = (𝜎2 + 𝜂2)𝛥𝑡𝑗 + 2𝜎𝜖2 
From 𝛺, we need to find the estimates of 𝜂 and 𝜎. 
But to solve the problem of ‘‘identifiability’’ is to estimate the parameters (𝜂 and 𝜎) with measurements sampled at a different interval.

## 5.3 State estimation and RUL evaluation
Kalman  Filter
