from DataFrame_genrator import *

df = generate_Dataframe_vector_xyz('data')
df = convert_to_columns_xyz(df)
df = convert_geodetic(df)
df = convert_to_degrees(df)
df = convert_to_mm_NE(df)

# columnsL: Date	Station	Position	Covariance	X	Y	Z	Latitude	Longitude	Height	Error	Latitude deg	Longitude deg	Distance_North_mm	Distance_East_mm


save_pickle_Dataframe(df, r'all_columns.pkl')
save_csv_Dataframe(df, r'all_columns.csv')