from predictions.earthquake_prediction.earthquake import EarthquakePredictionModel




if __name__ == "__main__":
    predictor = EarthquakePredictionModel("PHUK", "Thailand")
    results = predictor.run_complete_analysis()

