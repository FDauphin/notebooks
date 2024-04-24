# Algorithm Notebooks

Various Jupyter notebooks for learning new algorithms. A majority of the ML 
implementations use MNIST. 

Any applicable credits are within the notebooks, scripts, and modules. Includes:

- Generative Adversarial Networks
    - Deep Convolutional
    - Conditional Wasserstein with Gradient Penalty
    - Inversion
- Gaussian Process Pregression
    - Carbon dioxide data set
    - General tutorial with derivations
    - xsinx using scikit learn
    - xsinx using MCMC
- Miscellaneous
    - Self organizing maps
    - Symbolic regression with Newtonian gravity
    - Stochastic gradient descent on linear regression
    - Principal component analysis on double gaussians
    - Kernel density estimation
- Tensorflow
    - ResNet classification
    - Time series regression
- Time Series Analysis
    - Dimensionality Reduction
        - Functional principal component analysis on temperature data
        - Functional principal component analysis on height data
        - Principal component analysis on signals with oscillating amplitudes
    - Forecasting
        - ARIMA on airline passengers dataset
        - Exponential Smoothing on airline passengers dataset
        - Comparing ARIMA and exponential smoothing
    - Signal Processing
        - Fast Fourier Transform
        - Lomb Scargle Periodograms
        - Phase folding

## Installations

There are three environments: one purely conda (`environment_conda.yml`; only 
pip libraries are `tensorflow-macos`, `tensorflow-metal` and `minisom` since 
they do not have conda support), one purely pip (`requirements.txt`; only conda 
library needed is `scikit-fda` due to incompatibility in pip), and one mix 
(`environment.yml`; preferred). All notebooks were tested in the preferred 
environment using a 2021 Mac M1 chip.

```
# create preferred env
conda env create -f environment.yml

# create conda env
conda env create -f environment_conda.yml

# create pip env
conda create -n home_pip python=3.11 pip
pip install -r requirements.txt
conda install -c conda-forge scikit-fda
```

## Comparing models
- Plot data (train and test) vs model (train and test)
- Residuals plot and histogram
- Plot y_test vs y_pred
- MSE, MAE, and Pearson correlation