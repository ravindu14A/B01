from preprocess import preprocess
import pandas as pd

df = preprocess.generatePreprocessedDF()


pd.to_pickle(df, r"output\preprocessed.pkl")