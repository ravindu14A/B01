from sklearn.decomposition import PCA
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

def yohooo():
    # Save current working directory
    original_cwd = os.getcwd()

    # Get the path of the current script
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    os.chdir(script_dir)

    country = "Malaysia"
    directory = f"../processed_data/{country}/Filtered_cm_normalised"


    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            filepath = os.path.join(directory, filename)

            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                df = pd.DataFrame(data)

                X = df[['lat', 'long']].to_numpy()

                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)

                # Subtract the first point from all points
                X_pca = X_pca - X_pca[0]


                explained_variance = pca.explained_variance_ratio_

                out_df = pd.DataFrame({
                    'date': df['date'],
                    'lat': X_pca[:,0]
                })

            if filename == "ARAU.pkl":
                os.chdir(original_cwd)
                return pca
