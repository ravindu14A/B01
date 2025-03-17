import os
import re
import pandas as pd
from FileReader import GetDataFrame

def get_filenames(directory):
    pattern = re.compile(r'^PFITRF14.*\..*C$')
    filenames = [f for f in os.listdir(directory) if pattern.match(f)]
    return filenames

data_directory = 'data'
filenames = get_filenames(data_directory)

df = pd.DataFrame(columns=["Date", "Station", "Position", "Covariance"])
for i in filenames:
    newdf = GetDataFrame(data_directory+ r'\\' + i)
    df = pd.concat([df, newdf])

df.to_csv('output.csv', index=False)