from sklearn.decomposition import PCA
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

"""
Projecting the displacement into one direction
returns PCA model fitted to data
"""


def pca_fitting():
    # Save current working directory
    original_cwd = os.getcwd()

    # Get the path of the current script
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    start_date = pd.to_datetime('2004-12-30')
    end_date = pd.to_datetime('2026-12-30')

    os.chdir(script_dir)

    country = "Thailand"
    #directory = f"../data/partially_processed/{country}/Filtered_cm"
    directory = r"C:\Users\Laura\PycharmProjects_Uni\B01\data\partially_processed\Thailand\Filtered_cm"

    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            filepath = os.path.join(directory, filename)

            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                df = pd.DataFrame(data)

                # Filter by date
                df_subset = df[(df['date'] > start_date) & (df['date'] < end_date)]

                # Prepare training and full data
                X_train = df_subset[['lat', 'long']].to_numpy()
                X_whatever = df[['lat', 'long']].to_numpy()

                # Fit and transform
                pca = PCA(n_components=2)
                pca.fit(X_whatever)

            if filename == "PHUK.pkl":
                os.chdir(original_cwd)
                return pca
