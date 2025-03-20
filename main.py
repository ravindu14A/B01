import DataFrame_genrator
import pandas as pd
def SavePickleData():
    df = DataFrame_genrator.generate_Dataframe_vector_xyz('data')

    #filtered_sorted_df = df[df['Station'] == "BABH"].sort_values(by="Date")



    #print(filtered_sorted_df)
    df = DataFrame_genrator.convert_to_columns_xyz(df)

    df = DataFrame_genrator.convert_geodetic(df)

    DataFrame_genrator.save_pickle_Dataframe(df, r'pickle_geodetic_data_columns_xyz.pkl')


def GetSavedData():
    df = pd.read_pickle(r'output\\pickle_geodetic_data_columns_xyz.pkl')

    #print(df)
    return df

import pandas as pd

# Count the number of entries for each station
df = GetSavedData()
station_counts = df.groupby('Station').size()
station_counts_df = station_counts.reset_index(name='Count')

# Sort the stations by count in descending order
station_counts_sorted = station_counts_df.sort_values(by='Count', ascending=False)

# Print the sorted result
print(station_counts_sorted)
