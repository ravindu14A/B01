
from sklearn.linear_model import LinearRegression
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
stations = ['LGKW', 'GET2', 'USMP', 'KUAL', 'PUPK', 'PEKN', 'MERU', 'MRSG', 'JUML', 'SDLI', 'JHJY', 'KUKP', 'UMAS', 'BIN1', 'MIRI', 'LAB1', 'UMSS', 'KUDA', 'SAND','DATU']

slope = []
country = "Malaysia"
directory = f"../processed_data/{country}/Raw_pickle"


for filename in stations:
        filepath = directory + f"/{filename}.pkl"

        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            df = pd.DataFrame(data)

            df['date'] = pd.to_datetime(df['date'])

            # Convert dates to numeric values (e.g., days since start)
            X = (df['date'] - df['date'].min()).dt.total_seconds().values.reshape(-1, 1)
            y = df['alt'].values

            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)
            slp = model.coef_
            slope.append(slp*60*60*24*100) #cm/day

            # Predict values for the regression line
            y_pred = model.predict(X)

            # Plot
            # plt.figure(figsize=(12, 4))
            # plt.plot(df['date'], y, color='red', label='Altitude')
            # plt.plot(df['date'], y_pred, color='black', linestyle='--', label='Linear Trend')
            # plt.xlabel('Date')
            # plt.ylabel('Altitude')
            # plt.title('Linear Regression of Altitude Over Time')
            # plt.legend()
            # plt.grid(True)
            # plt.tight_layout()
            # plt.show()


for i,j in zip(stations, slope):
    print(f"Vertical motion of {i}: {j} cm/day")