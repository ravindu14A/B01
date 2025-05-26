import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, bisect


class EarthquakePredictionModel:
    """
    A class for earthquake prediction based on latitude changes over time.

    This model fits a linear regression to pre-earthquake data and then uses
    an exponential decay model to predict when the latitude will return to
    its initial value, potentially indicating an earthquake event.
    """

    def __init__(self, station, country):
        """
        Initialize the earthquake prediction model.

        Args:
            station (str): Station identifier
            country (str): Country name
        """
        self.station = station
        self.country = country
        self.df = None
        self.model = None
        self.slope_per_day = None
        self.popt = None
        self.start_date = None
        self.end_date = None
        self.start_day1 = None
        self.df_subset = None
        self.df_fit = None

    def load_data(self, filepath=None):
        """
        Load data from pickle file.

        Args:
            filepath (str, optional): Custom filepath. If None, uses default pattern.
        """
        if filepath is None:
            filepath = f"../data/final/{self.country}/{self.station}.pkl"

        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.df = pd.DataFrame(data)

        # Process dates
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.start_date = self.df["date"].iloc[0]
        self.df['days_from_ref'] = (self.df['date'] - self.start_date).dt.days

        print(f"Data loaded successfully for {self.station}, {self.country}")
        print(f"Date range: {self.start_date} to {self.df['date'].iloc[-1]}")

    def fit_linear_model(self, end_date="2004-12-15"):
        """
        Fit linear regression model to the pre-earthquake period.

        Args:
            end_date (str): End date for linear fitting period
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.end_date = pd.to_datetime(end_date)

        # Create subset for linear fitting
        mask = (self.df['date'] >= self.start_date) & (self.df['date'] <= self.end_date)
        self.df_subset = self.df.loc[mask]

        X = self.df_subset['days_from_ref'].values.reshape(-1, 1)
        y = self.df_subset['lat'].values

        # Fit linear model
        self.model = LinearRegression()
        self.model.fit(X, y)
        self.slope_per_day = self.model.coef_[0]

        slope_mm_per_year = self.slope_per_day * 10 * 365
        print(f"Linear model fitted. Slope (mm/year): {slope_mm_per_year:.3f}")

    def model_func(self, t, A, B, c1, c2, d):
        """
        Exponential decay model function.

        Args:
            t: Time values
            A, B, c1, c2, d: Model parameters

        Returns:
            Model predictions
        """
        return (A * np.exp(-c1 * (t - self.start_day1)) +
                B * np.exp(-c2 * (t - self.start_day1)) +
                d + self.slope_per_day * (t - self.start_day1))

    def fit_exponential_model(self):
        """
        Fit exponential decay model to post-earthquake period.
        """
        if self.model is None:
            raise ValueError("Linear model not fitted. Call fit_linear_model() first.")

        # Get post-earthquake data
        self.df_fit = self.df.loc[self.df['date'] > self.end_date]
        self.start_day1 = self.df_fit["days_from_ref"].iloc[0]

        t_data = self.df_fit['days_from_ref'].values
        y_data = self.df_fit['lat'].values

        # Fit exponential model
        self.popt, pcov = curve_fit(self.model_func, t_data, y_data)
        A, B, c1, c2, d = self.popt

        param_names = ['A', 'B', 'c1', 'c2', 'd']
        print("Exponential model parameters:")
        for name, val in zip(param_names, self.popt):
            print(f"  {name} = {val:.6f}")

    def predict_earthquake(self, years=340):
        """
        Predict when the next earthquake might occur.

        Args:
            years (int): Number of years to project into the future

        Returns:
            tuple: (root_years, predicted_year) or (None, None) if no intersection found
        """
        if self.popt is None:
            raise ValueError("Exponential model not fitted. Call fit_exponential_model() first.")

        safe_end = round(years * 365.25)

        def f(t):
            return self.model_func(t, *self.popt) - self.df_fit["lat"].iloc[0]

        try:
            root = bisect(f, 10000, safe_end - 1)
            root_years = root / 365.25
            predicted_year = int(self.start_date.year + root / 365.25)

            print(f"Earthquake prediction: {root_years:.2f} years after {self.start_date}")
            print(f"Predicted year: ~{predicted_year}")

            return root_years, predicted_year
        except Exception as e:
            print(f"Could not find earthquake prediction: {e}")
            return None, None

    def plot_results(self, years=340, figsize=(12, 6)):
        """
        Plot the complete analysis results.

        Args:
            years (int): Number of years to project
            figsize (tuple): Figure size
        """
        if self.popt is None:
            raise ValueError("Models not fitted. Call fit_linear_model() and fit_exponential_model() first.")

        plt.figure(figsize=figsize)

        # Generate prediction data
        safe_end = round(years * 365.25)
        T = np.arange(self.start_day1, safe_end, 5)
        y_fit = self.model_func(T, *self.popt)
        ref = np.full_like(T, self.df_fit["lat"].iloc[0])

        # Convert to years for plotting
        T_years = T / 365.25
        df_years = self.df['days_from_ref'] / 365.25
        df_subset_years = self.df_subset['days_from_ref'] / 365.25

        # Linear fit predictions
        X = self.df_subset['days_from_ref'].values.reshape(-1, 1)
        y_pred = self.model.predict(X)

        # Plot earthquake prediction point if available
        root_years, predicted_year = self.predict_earthquake(years)
        if root_years is not None:
            y_point = self.model_func(root_years * 365.25, *self.popt)
            plt.scatter(root_years, y_point, color='blue', zorder=5, s=100)
            plt.text(root_years + 5, y_point - 2,
                     f"Predicted EQ\n~Year {predicted_year}", fontsize=10)

        # Plot all data
        plt.plot(T_years, y_fit, 'r-', label='Fitted Curve', linewidth=2)
        plt.plot(df_years, self.df['lat'], label='Original Data', color='green', alpha=0.7)
        plt.plot(df_subset_years, y_pred, color='red', label='Linear Fit Segment', linewidth=2)
        plt.plot(T_years, ref, color='black', linestyle="--", label='Initial Lat', alpha=0.8)

        plt.xlabel('Years since reference date')
        plt.ylabel('Latitude (PCA transformed)')
        plt.title(f'{self.station} - Earthquake Prediction (Relative Time)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def run_complete_analysis(self, filepath=None, end_date="2004-12-15",
                              prediction_years=340, plot=True):
        """
        Run the complete earthquake prediction analysis.

        Args:
            filepath (str, optional): Data file path
            end_date (str): End date for linear fitting
            prediction_years (int): Years to project for prediction
            plot (bool): Whether to generate plot
        """
        print(f"Starting earthquake prediction analysis for {self.station}, {self.country}")
        print("=" * 60)

        # Load and process data
        self.load_data(filepath)

        # Fit models
        self.fit_linear_model(end_date)
        self.fit_exponential_model()

        # Make prediction
        root_years, predicted_year = self.predict_earthquake(prediction_years)

        # Plot results
        if plot:
            self.plot_results(prediction_years)

        print("=" * 60)
        print("Analysis complete!")

        return {
            'slope_mm_per_year': self.slope_per_day * 10 * 365,
            'model_parameters': dict(zip(['A', 'B', 'c1', 'c2', 'd'], self.popt)),
            'prediction_years': root_years,
            'predicted_year': predicted_year
        }