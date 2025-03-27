from util.load_dataset import load_geodataset
from  missing_data.filling_missing_data import MissingDataGNSS
from outlier_detection.outlier import OutlierDetector
from internal_dataclass.dataset import GeoDataset
def main():
    data_path = 'processed_data\SE_Asia'

    dataset = load_geodataset(data_path)

    #If u want to use only one sample
    #dataset = GeoDataset(samples=dataset.samples[:1])

    filling_data = MissingDataGNSS(dataset)
    dataset = filling_data.processing_all_files()

    """outlied_detector = OutlierDetector(dataset)
    dataset = outlied_detector.clean_dataset()

    print(dataset.samples)"""



if __name__ == '__main__':
    main()