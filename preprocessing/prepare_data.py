from util.load_dataset import load_geodataset
from  missing_data.filling_missing_data import MissingDataGNSS


def main():
    data_path = 'processed_data\SE_Asia'

    dataset = load_geodataset(data_path)
    filling_data = MissingDataGNSS(dataset)
    dataset = filling_data.processing_all_files()


if __name__ == '__main__':
    main()