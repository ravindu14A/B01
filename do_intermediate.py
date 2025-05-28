from intermediate import velocity_correction
from intermediate import center_on_eq
import pandas as pd

df = pd.read_pickle(r'output\preprocessed.pkl')

df = velocity_correction.apply_velocity_correction(df)
df = center_on_eq.center_days_since_eq(df)# adds column days_since_eq ()
#df = center_on_eq.decades_since_eq(df) # adds column decades_since_eq

# linear trend is used to set the earthquake date as 0,0 point in north and east.
# TO DO!!! if there is not enough points before eq (by default 5), it takes the first point available as 0, change that
# if useful this will be useful for a much fuller model

df = center_on_eq.center_all_stations_with_trend(df) #


pd.to_pickle(df,r"output\intermediate.pkl")
