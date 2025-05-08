import pickle
import pandas as pd
from sklearn.decomposition import PCA
from preprocessing.downloading_data.PCA import yohooo
import numpy as np


# Replace 'your_file.pkl' with the actual file path
with open(r'C:\Users\Laura\PycharmProjects_Uni\B01\preprocessing\processed_data\Malaysia\PCA\ARAU.pkl', 'rb') as file:
    data = pickle.load(file)

# Now 'data' contains the object that was saved in the pickle file
with open(r'C:\Users\Laura\PycharmProjects_Uni\B01\preprocessing\processed_data\Malaysia\Filtered\ARAU.pkl', 'rb') as file:
    cov_ARAU = pickle.load(file)

merged_df = pd.merge(data, cov_ARAU[['date', 'covariance matrix']], on='date', how='inner')  # Options: 'inner', 'outer', 'left', 'right'


print(merged_df) # north-sout, east-west

def new_covariance(matrix):
    """
    Args:
        matrix: 3x3 matrix north, east, up

    Returns: single value of interest
    """
    slic_matrix = matrix[:2, :2]
    pca = yohooo()
    pca_var = pca.components_[0]
    return pca_var @ slic_matrix @ pca_var.T

def smart_epsilon(x0):
    eps_machine = np.finfo(float).eps
    return np.sqrt(eps_machine) * np.maximum(np.abs(x0), 1.0)


def jacobian(x0,f):
    n = len(x0)
    m = len(f(x0))
    J = np.zeros((m, n))
    eps_vec = smart_epsilon(x0)

    for i in range(n):
        dx = np.zeros_like(x0)
        dx[i] = eps_vec[i]
        f_plus = f(x0 + dx)
        f_minus = f(x0 - dx)
        J[:, i] = (f_plus - f_minus) / (2 * dx[i])

    return J

merged_df['covariance matrix'] = merged_df.apply(lambda row:new_covariance(row['covariance matrix']), axis=1)

print(merged_df)