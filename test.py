import pandas as pd


df = pd.read_pickle(r"output\velocity_corrected.pkl")



print(df.columns)


# filtered_df = df[(df['date'] == pd.Timestamp('2000-01-04')) & (df['station'] == 'NTUS')]
# phkt = df[df["station"]=='PHKT']
# phuk = df[df["station"]=='PHUK']

# print(phuk.iloc[0])
# print(phkt.iloc[0])

# print(filtered_df[[ 'x', 'y', 'z', 'd_north_mm', 'd_east_mm', 'd_up_mm']])











from geo_utils import geo_utils
#geo_utils.displacement()



import matplotlib.pyplot as plt

def plot_displacement_for_station(df, station_name):
    # Filter DataFrame by the station name
    station_data = df[df['station'] == station_name]
    
    # Ensure the 'date' column is in datetime format
    station_data['date'] = pd.to_datetime(station_data['date'])
    
    # Sort the data by date in chronological order
    station_data = station_data.sort_values(by='date')

    # Plot North displacement (d_north_mm)
    plt.figure(figsize=(10, 6))
    plt.plot(station_data['date'], station_data['d_north_mm'], label='North Displacement (mm)', color='b', linewidth=1)
    plt.xlabel('Date')
    plt.ylabel('North Displacement (mm)')
    plt.title(f"North Displacement for Station {station_name}")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot East displacement (d_east_mm)
    plt.figure(figsize=(10, 6))
    plt.plot(station_data['date'], station_data['d_east_mm'], label='East Displacement (mm)', color='g', linewidth=1)
    plt.xlabel('Date')
    plt.ylabel('East Displacement (mm)')
    plt.title(f"East Displacement for Station {station_name}")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot Height displacement (d_up_mm)
    plt.figure(figsize=(10, 6))
    plt.plot(station_data['date'], station_data['d_up_mm'], label='Height (mm)', color='r', linewidth=1)
    plt.xlabel('Date')
    plt.ylabel('Height (mm)')
    plt.title(f"Height for Station {station_name}")
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage
#plot_displacement_for_station(df, "ARAU")  # Replace "NTUS" with your desired station name
