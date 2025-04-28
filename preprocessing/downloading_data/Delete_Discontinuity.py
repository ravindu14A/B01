import numpy as np
import pandas as pd
import os
import pickle
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression

country = "Thailand"

begin_time = pd.to_datetime("2010-01-01")
earth_time1 = pd.to_datetime("2012-01-01")
earth_time2 = pd.to_datetime("2013-01-01")
end_time = pd.to_datetime("2015-01-01")

#2012
directory = f"../processed_data/{country}/PCA"

directory_out = f"../processed_data/{country}/Final"

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
            date = df["date"].to_numpy()
            response = df["lat"].to_numpy()
            print(date.shape)
            # model = LinearRegression()
            #
            # m1 = model.fit()

    filename = os.path.join(directory_out, f"{filename}")
    df.to_pickle(filename)
