from sklearn.decomposition import PCA
import os
import pickle
import pandas as pd
# import main as mn

# Load data
country = "Malaysia"

directory = f"../processed_data/{country}/Filtered_cm_normalised"
directory_out = f"../processed_data/{country}/PCA"
start_date = pd.to_datetime("2004-12-30")

trans = {}
# Clear the output directory
for filename in os.listdir(directory_out):
    file_path = os.path.join(directory_out, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Process each file
for filename in os.listdir(directory):
    if filename.endswith('.pkl'):
        filepath = os.path.join(directory, filename)

        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            df = pd.DataFrame(data)

        # Filter by date
        df_subset = df[df['date'] >= start_date]
        if df_subset.empty:
            print(f"Skipping {filename}: no data after {start_date.date()}")
            continue

        # Prepare training and full data
        X_train = df_subset[['lat', 'long']].to_numpy()
        X_all = df[['lat', 'long']].to_numpy()

        # Fit and transform
        pca = PCA(n_components=2)
        pca.fit(X_train)
        X_pca = pca.transform(X_all)

        P = pca.components_

        trans[f"{filename[:-4]}"] = P

        # Save output
        out_df = pd.DataFrame({
            'date': df['date'].values,
            'lat': X_pca[:, 0],
            'long': X_pca[:, 1]
        })

        out_path = os.path.join(directory_out, filename)
        out_df.to_pickle(out_path)

