from sklearn.decomposition import PCA
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt


country = "Thailand"
directory = f"../processed_data/{country}/Filtered_cm_normalised"

directory_out = f"../processed_data/{country}/PCA"

for filename in os.listdir(directory_out):
    file_path = os.path.join(directory_out, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

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

    filename = os.path.join(directory_out, f"{filename}")
    out_df.to_pickle(filename)