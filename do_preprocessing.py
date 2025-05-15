from preprocess.preprocess import *
import pandas as pd

df = readAllFiles()
removeDuplicates(df)
addLatLonHeight(df)
addRelativeDisplacementmm(df)
convert_xyz_cov_to_enu(df)


pd.to_pickle(df, r"output\preprocessed.pkl")