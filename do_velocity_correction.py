from analysis import velocity_correction
import pandas as pd

df = pd.read_pickle(r'output\preprocessed.pkl')
df = velocity_correction.apply_velocity_correction(df)
pd.to_pickle(df,r"output\velocity_corrected.pkl")
