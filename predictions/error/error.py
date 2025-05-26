import pickle
import pandas as pd
from preprocessing.PCA.PCA import pca_fitting
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from typing import Tuple


class TimeSeriesForecaster:
    """
    A class for time series analysis with PCA-based covariance processing
    and Monte Carlo forecasting using exponential decay models.
    """

    def __init__(self, pca_file_path: str, covariance_file_path: str):
        """
        Initialize the forecaster with data file paths.

        Args:
            pca_file_path: Path to the PCA pickle file
            covariance_file_path: Path to the covariance matrix pickle file
        """
        self.pca_file_path = pca_file_path
        self.covariance_file_path = covariance_file_path
        self.merged_df = None
        self.slope_per_day = None
        self.simulated_datasets = None

    def load_data(self) -> pd.DataFrame:
        """Load and merge the PCA and covariance data."""
        # Load PCA data
        with open(self.pca_file_path, 'rb') as file:
            pca_data = pickle.load(file)

        # Load covariance data
        with open(self.covariance_file_path, 'rb') as file:
            cov_data = pickle.load(file)

        # Merge datasets
        self.merged_df = pd.merge(
            pca_data,
            cov_data[['date', 'covariance matrix']],
            on='date',
            how='inner'
        )

        return self.merged_df

    @staticmethod
    def new_covariance(matrix: np.ndarray) -> float:
        """
        Process covariance matrix using PCA.

        Args:
            matrix: 3x3 matrix (north, east, up)

        Returns:
            Single value of interest after PCA processing
        """
        slic_matrix = matrix[:2, :2]
        pca = pca_fitting()
        pca_var = pca.components_[0]
        result = pca_var @ slic_matrix @ pca_var.T
        print(result, pca_var)
        return result

    @staticmethod
    def smart_epsilon(x0: np.ndarray) -> np.ndarray:
        """Calculate smart epsilon for numerical stability."""
        eps_machine = np.finfo(float).eps
        return np.sqrt(eps_machine) * np.maximum(np.abs(x0), 1.0)

    def process_covariance_matrices(self):
        """Process all covariance matrices in the dataset."""
        if self.merged_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("Sample covariance matrix:")
        print(self.merged_df.iloc[0]['covariance matrix'])

        self.merged_df['covariance matrix'] = self.merged_df.apply(
            lambda row: self.new_covariance(row['covariance matrix']),
            axis=1
        )

    def prepare_time_data(self, end_date: str = "2004-12-15"):
        """
        Prepare time-based features for analysis.

        Args:
            end_date: End date for slope estimation subset
        """
        if self.merged_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Convert date to datetime
        self.merged_df['date'] = pd.to_datetime(self.merged_df['date'])

        # Set reference dates
        start_date = self.merged_df['date'].iloc[0]
        end_date = pd.to_datetime(end_date)

        # Add numeric time axis
        self.merged_df['days_from_ref'] = (self.merged_df['date'] - start_date).dt.days

        return start_date, end_date

    def estimate_slope(self, end_date: str = "2004-12-15") -> float:
        """
        Estimate the linear slope using data up to the specified end date.

        Args:
            end_date: End date for slope estimation

        Returns:
            Slope per day
        """
        if self.merged_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        end_date = pd.to_datetime(end_date)

        # Subset data for slope estimation
        df_subset = self.merged_df[self.merged_df['date'] <= end_date]

        # Prepare features and target
        X = df_subset['days_from_ref'].values.reshape(-1, 1)
        y = df_subset['lat'].values

        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        self.slope_per_day = model.coef_[0]

        # Convert to mm/year for display
        slope_mm_per_year = self.slope_per_day * 10 * 365
        print(f"Slope (mm/year): {slope_mm_per_year:.3f}")

        return self.slope_per_day

    def prepare_monte_carlo(self, start_index: int = 350, n_samples: int = 50):
        """
        Prepare Monte Carlo simulation datasets.

        Args:
            start_index: Starting index for data subset
            n_samples: Number of Monte Carlo samples
        """
        if self.merged_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Subset data
        self.merged_df = self.merged_df.iloc[start_index:]

        # Extract diagonal values and create covariance matrix
        diag_vals = self.merged_df['covariance matrix'].values.tolist()
        D = np.diag(diag_vals) * 10000

        print("Covariance matrix shape:", D.shape)

        # Get observed values
        y = self.merged_df['lat'].values

        # Generate simulated datasets
        self.simulated_datasets = np.random.multivariate_normal(y, D, size=n_samples)

        return self.simulated_datasets

    def model_func(self, t: np.ndarray, A: float, B: float, c1: float, c2: float, d: float) -> np.ndarray:
        """
        Exponential decay model with linear trend.

        Args:
            t: Time array
            A, B: Exponential coefficients
            c1, c2: Decay constants
            d: Offset

        Returns:
            Model predictions
        """
        if self.slope_per_day is None:
            raise ValueError("Slope not estimated. Call estimate_slope() first.")

        t_base = self.get_time_numeric()[0]  # First time point
        return (A * np.exp(-c1 * (t - t_base)) +
                B * np.exp(-c2 * (t - t_base)) +
                d + self.slope_per_day * (t - t_base))

    def get_time_numeric(self) -> np.ndarray:
        """Convert dates to numeric time (years since first date)."""
        if self.merged_df is None:
            raise ValueError("Data not loaded.")

        dates = self.merged_df['date'].to_list()
        base_date = dates[0]
        return np.array([(d - base_date).days for d in dates])

    def forecast_and_plot(self, forecast_years: int = 30, figsize: Tuple[int, int] = (12, 6)):
        """
        Generate forecasts using Monte Carlo simulation and plot results.

        Args:
            forecast_years: Number of years to forecast
            figsize: Figure size for plotting
        """
        if self.simulated_datasets is None:
            raise ValueError("Monte Carlo data not prepared. Call prepare_monte_carlo() first.")

        plt.figure(figsize=figsize)

        # Get time data
        t_numeric = self.get_time_numeric()
        values = self.merged_df['lat'].values

        # Generate forecasts for each Monte Carlo sample
        N = len(self.simulated_datasets)

        for i in range(N):
            try:
                # Fit model to simulated data
                params, _ = curve_fit(self.model_func, t_numeric, self.simulated_datasets[i])

                # Create future time points
                future_days = forecast_years * 365
                t_future_numeric = np.arange(t_numeric[-1] + 1, t_numeric[-1] + future_days + 1)

                # Generate predictions
                future_preds = self.model_func(t_future_numeric, *params)

                # Plot (only show label for first forecast line)
                label = '30-Year Forecast' if i == 0 else None
                alpha = 0.1 if N > 10 else 0.3
                plt.plot(t_future_numeric, future_preds,
                         color='red', alpha=alpha, label=label)

            except Exception as e:
                print(f"Warning: Failed to fit sample {i}: {e}")
                continue

        # Plot observed data
        plt.plot(t_numeric, values, label='Observed', color='blue', linewidth=2)

        # Formatting
        plt.xlabel("Years since start")
        plt.ylabel("Latitude")
        plt.title(f"Exponential Decay Model with {forecast_years}-Year Monte Carlo Forecast")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def run_full_analysis(self,
                          end_date: str = "2004-12-15",
                          start_index: int = 350,
                          n_samples: int = 50,
                          forecast_years: int = 30):
        """
        Run the complete analysis pipeline.

        Args:
            end_date: End date for slope estimation
            start_index: Starting index for Monte Carlo data
            n_samples: Number of Monte Carlo samples
            forecast_years: Years to forecast
        """
        print("Loading data...")
        self.load_data()

        print("Processing covariance matrices...")
        self.process_covariance_matrices()

        print("Preparing time data...")
        self.prepare_time_data(end_date)

        print("Estimating slope...")
        self.estimate_slope(end_date)

        print("Preparing Monte Carlo simulation...")
        self.prepare_monte_carlo(start_index, n_samples)

        print("Generating forecasts and plotting...")
        self.forecast_and_plot(forecast_years)

        print("Analysis complete!")


# Example usage:
if __name__ == "__main__":
    # Initialize forecaster
    forecaster = TimeSeriesForecaster(
        pca_file_path=r'../../data/partially_processed/Thailand/PCA/PHUK.pkl',
        covariance_file_path=r'../../data/partially_processed/Thailand/Filtered_cm/PHUK.pkl'
    )

    # Run complete analysis
    forecaster.run_full_analysis(
        end_date="2004-12-15",
        start_index=350,
        n_samples=50,
        forecast_years=400
    )