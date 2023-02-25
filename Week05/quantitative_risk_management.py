import numpy as np
import pandas as pd
from scipy.linalg import cholesky, eigvals
from scipy import stats
from scipy.optimize import minimize 
from statsmodels.tsa.arima.model import ARIMA

#exponentially weighted covaraince matrix
def exp_weighted_cov(input, lambda_=0.97):
    ror = input.values
    ror_mean = np.mean(ror, axis=0)
    dev = ror - ror_mean
    times = dev.shape[0]
    weights = np.zeros(times)
    
    for i in range(times):
        weights[times - i - 1]  = (1 - lambda_) * lambda_**i
    
    weights_mat = np.diag(weights/sum(weights))

    cov = np.transpose(dev) @ weights_mat @ dev
    return cov

#simulate pca
def simulate_pca(a, nsim, nval=None):
    # Compute the eigenvalues and eigenvectors of a.
    vals, vecs = np.linalg.eig(a)

    # Sort the eigenvalues and eigenvectors in descending order of eigenvalue.
    idx = np.argsort(-vals)
    vals = vals[idx]
    vecs = vecs[:, idx]

    # Keep only the non-negative eigenvalues.
    posv = np.where(vals >= 1e-8)[0]
    if nval is not None:
        # If nval is specified, keep only the first nval non-negative eigenvalues.
        if nval < len(posv):
            posv = posv[:nval]
    vals = vals[posv]
    vecs = vecs[:, posv]

    # Compute the cumulative explained variance of the first nsim principal components.
    cum_var = (np.cumsum(vals[:nsim]) / np.sum(vals))[-1]

    # Return the cumulative explained variance.
    return cum_var

#Frobenius norm
def Frobenius(input):
    result = 0
    for i in range(len(input)):
        for j in range(len(input)):
            result += input[i][j]**2
    return result


#Higham psd
def Higham_psd(input):
    weight = np.identity(len(input))
        
    norml = np.inf
    Yk = input.copy()
    Delta_S = np.zeros_like(Yk)
    
    invSD = None
    if np.count_nonzero(np.diag(Yk) == 1.0) != input.shape[0]:
        invSD = np.diag(1 / np.sqrt(np.diag(Yk)))
        Yk = invSD @ Yk @ invSD
    
    Y0 = Yk.copy()

    for i in range(1000):
        Rk = Yk - Delta_S
        # PS
        Xk = np.sqrt(weight)@ Rk @np.sqrt(weight)
        vals, vecs = np.linalg.eigh(Xk)
        vals = np.where(vals > 0, vals, 0)
        Xk = np.sqrt(weight)@ vecs @ np.diagflat(vals) @ vecs.T @ np.sqrt(weight)
        Delta_S = Xk - Rk
        #PU
        Yk = Xk.copy()
        np.fill_diagonal(Yk, 1)
        norm = Frobenius(Yk-Y0)
        #norm = np.linalg.norm(Yk-Y0, ord='fro')
        min_val = np.real(np.linalg.eigvals(Yk)).min()
        if abs(norm - norml) < 1e-8 and min_val > -1e-9:
            break
        else:
            norml = norm
    
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        Yk = invSD @ Yk @ invSD
    return Yk

#cholesky fix matrix
def chol_psd(a):
    n = a.shape[0]
    root = np.zeros((n, n))

    for j in range(n):
        # Calculate the squared norm of the previous elements in the j-th row of root.
        s = 0.0
        if j > 0:
            s = np.dot(root[j, :j], root[j, :j])
        
        # Compute the j-th diagonal element of root.
        temp = a[j, j] - s
        if temp <= 0:
            temp = 0.0
        root[j, j] = np.sqrt(temp)

        # If the j-th diagonal element of root is zero, set the remaining elements in the j-th row to zero.
        if root[j, j] == 0.0:
            root[j, j+1:] = 0.0
        # Otherwise, compute the elements below the diagonal in the j-th column of root.
        else:
            ir = 1.0 / root[j, j]
            for i in range(j + 1, n):
                s = np.dot(root[i, :j], root[j, :j])
                root[i, j] = (a[i, j] - s) * ir
    
    # Return the lower triangular matrix root.
    return root
#or apply:cholesky_fixed_matrix(A), A is a matrix(numpy 2D array)

#near psd
def near_psd(a, epsilon=0.0):
    n = a.shape[0]

    invSD = None
    out = a.copy()

    # calculate the correlation matrix if we got a covariance
    if np.count_nonzero(np.diag(out) == 1.0) != n:
        invSD = np.diag(1 / np.sqrt(np.diag(out)))
        out = np.matmul(np.matmul(invSD, out), invSD)

    # SVD, update the eigen value and scale
    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = np.reciprocal(np.matmul(np.square(vecs), vals))
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T

    #Add back the variance
    if invSD is not None:
        invSD = np.diag(1 / np.diag(invSD))
        out = invSD @ out @ invSD
    return out


#simulate multivariate normal
def multivar_norm_simu(cov, method='direct', mean = 0, explained_variance=1.0, samples_num=25000):
    if method == 'direct':
        L = chol_psd(cov)
        normal_samples = np.random.normal(size=(cov.shape[0], samples_num))
        samples = np.transpose(L @ normal_samples + mean)
        return samples
    
    elif method == 'pca':
        vals, vecs = np.linalg.eigh(cov)
        idx = vals > 1e-8
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        if explained_variance == 1.0:
            explained_variance = (np.cumsum(vals)/np.sum(vals))[-1]
        
        n_components = np.where((np.cumsum(vals)/np.sum(vals))>= explained_variance)[0][0] + 1
        vecs = vecs[:,:n_components]
        vals = vals[:n_components]

        normal_samples = np.random.normal(size=(n_components, samples_num))
        
        B = vecs @ np.diag(np.sqrt(vals))
        samples = np.transpose(B @ normal_samples)
        
        return samples
    
#Pearson correlation and EW variance
def p_corr_ew_var_f(returns, lambda_=0.97):
    cov = exp_weighted_cov(returns, lambda_)
    std_dev = np.sqrt(np.diag(cov))
    corr = np.corrcoef(returns.T)
    cov = np.outer(std_dev, std_dev) * corr
    return cov

#Delta Norm VaR
def cal_delta_VaR(Total_Value, assets_prices, delta, alpha=0.05, lambda_=0.94):
    returns = assets_prices.pct_change()
    assets_cov = exp_weighted_cov(returns, lambda_)
    
    delta_norm_VaR = -Total_Value * stats.norm.ppf(alpha) * np.sqrt(delta.T @ assets_cov @ delta)
    
    return delta_norm_VaR.item()

# Monte Carlo VaR
def calculate_MC_var(assets_prices, holdings, alpha=0.05, lambda_=0.94, n_simulation = 10000):
    returns = assets_prices.pct_change()
    returns_norm = returns - returns.mean()
    assets_cov = exp_weighted_cov(returns_norm, lambda_)
    np.random.seed(0)
    simu_returns = np.add(multivar_norm_simu(assets_cov), returns.mean().values)
    print(simu_returns)
    simu_prices = np.dot(simu_returns* assets_prices.tail(1).values.reshape(assets_prices.shape[1],),holdings)
    MC_VaR = -np.percentile(simu_prices, alpha*100)
    return MC_VaR

#Historical VaR
def cal_hist_VaR(assets_prices, holdings, alpha=0.05):
    returns = assets_prices.pct_change()
    simu_returns = returns.sample(1000, replace=True)
    simu_prices = np.dot(simu_returns* assets_prices.tail(1).values.reshape(assets_prices.shape[1],),holdings)

    hist_VaR = -np.percentile(simu_prices, alpha*100)

    return hist_VaR