import numpy as np
import pandas as pd
import os
import pickle
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Load data
country = "malaysia"


directory = f"../data/partially_processed_steps/{country}/pca"

directory_out = f"../data/processed/{country}"

def correct_discontinuities(df, column='lat', threshold=5, start_date=None, max_days=1):

    corrected = df.copy()
    shifts = np.zeros(len(corrected))
    dates = corrected['date']
    if dates.max() > pd.to_datetime(start_date):
        if start_date is not None:
            start_index = corrected[dates >= pd.to_datetime(start_date)].index[0]

        else:
            start_index = 1

        for i in range(start_index, len(corrected)):
            delta_val = corrected[column].iloc[i] - corrected[column].iloc[i - 1]
            delta_time = (dates.iloc[i] - dates.iloc[i - 1]).days

            if abs(delta_val) > threshold and delta_time <= max_days:
                shifts[i:] -= delta_val

        corrected[column] = corrected[column] + shifts
    return corrected

t_treshold = [3,0.8,1,1,1,1,2,1,1,1,1,1,1,1,1,1]
t_max = [10,100,5,200,10,5,350,5,100,200,5,5,5,5,50,1]

m_treshold = [3,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
m_max = [10,100,5,200,10,5,10,30,5,1,5,5,100,5,50,100,100]
if country == "malaysia":
    treshold = m_treshold
    max = m_max
elif country == "thailand":
    treshold = t_treshold
    max = t_max
    
for num, filename in enumerate(os.listdir(directory)):
    if filename.endswith('.pkl'):
        filepath = os.path.join(directory, filename)


        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            df = pd.DataFrame(data)
            corrected = correct_discontinuities(df, column='lat', threshold=treshold[num], start_date="2011-12-01", max_days=max[num])

            plt.figure(figsize=(12, 5))
            plt.plot(df['date'], df['lat'], label='Original', color='red', alpha=0.6)
            plt.plot(corrected['date'], corrected['lat'], label='Corrected', color='green')
            plt.title(f'Discontinuity Correction in North-South Displacement for {filename}')
            plt.xlabel('Date')
            plt.ylabel('North-South (cm)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        filename = os.path.join(directory_out, f"{filename}")
        corrected.to_pickle(filename)










