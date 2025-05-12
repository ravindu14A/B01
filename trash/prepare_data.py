from util.load_dataset import load_geodataset
from  missing_data.filling_missing_data import MissingDataGNSS
from internal_dataclass.dataset import GeoDataset
import pickle
import matplotlib.pyplot as plt
import pandas as pd

def save_data(ds, output_file = 'results'):
    with open(f'{output_file}.pkl', 'wb') as f:
        pickle.dump(ds, f)


def plot_station_columns(dataset: GeoDataset, station_name: str, x_col = 'date', y_col = 'long'):
    """
    Plots specific columns from a station's observation data.

    Args:
        dataset (GeoDataset): The dataset containing stations.
        station_name (str): Name of the station to plot.
        x_col (str): Column name to use for the x-axis.
        y_col (str): Column name to use for the y-axis.
    """
    # Find the station
    station = next((s for s in dataset.samples if s.name == station_name), None)

    if not station:
        print(f"Station '{station_name}' not found in dataset.")
        return

    df = station.data
    df.reset_index(inplace=True)

    df['date'] = pd.to_datetime(df['date'])
    print(df['date'])

    # Validate columns
    if x_col not in df.columns or y_col not in df.columns:
        print(f"Columns '{x_col}' or '{y_col}' not found in station '{station_name}' data.")
        return

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_col], df[y_col], marker='o')
    plt.title(f"{y_col} vs {x_col} for Station: {station.name}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    data_path = r'../data/partially_processed/Malaysia/Filtered_cm'

    dataset = load_geodataset(data_path)
    #If u want to use only one sample
    #dataset = GeoDataset(samples=dataset.samples[:1])
    dataset = GeoDataset(samples=dataset.samples)

    plot_station_columns(dataset, station_name='mm_KUAL')

    filling_data = MissingDataGNSS(dataset)
    dataset = filling_data.processing_all_files()

    '''outlied_detector = OutlierDetector(dataset)
    dataset = outlied_detector.clean_dataset()'''

    for sample in dataset.samples:
        if sample.name == 'mm_KUAL':
            with open('../predictions/KUAL_new.pkl', 'wb') as f:
                pickle.dump(sample, f)
    # Save the data as pickle
    save_data(dataset)
    plot_station_columns(dataset, station_name='mm_KUAL')

if __name__ == '__main__':
    main()