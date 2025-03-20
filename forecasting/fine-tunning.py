import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# 🔹 Generate a tiny synthetic time-series dataset
dates = pd.date_range(start="2023-01-01", periods=100, freq="D")  # 100 days
values = np.cumsum(np.random.randn(100) * 2) + 50  # Random walk

# 🔹 Create DataFrame
train_data = pd.DataFrame({"timestamp": dates, "target": values})
train_data["item_id"] = "synthetic_series"  # Required for AutoGluon
train_data = train_data.set_index(["item_id", "timestamp"])  # Set multi-index
train_data = TimeSeriesDataFrame(train_data)
print("📌 Sample Dataset:")
print(train_data.head())

# 🔹 Fine-Tune Chronos-T5-Small (bolt_small)
predictor = TimeSeriesPredictor(prediction_length=10).fit(
    train_data=train_data,
    hyperparameters={
        "Chronos": [
            {"model_path": "bolt_small", "ag_args": {"name_suffix": "ZeroShot"}},  # Pretrained model
            {"model_path": "bolt_small", "fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},  # Fine-tuned
        ]
    },
    time_limit=60,  # Training time in seconds
    enable_ensemble=False,  # No ensembling for faster results
)

print("✅ Fine-tuning completed!")

# 🔹 Evaluate performance
leaderboard = predictor.leaderboard(train_data)
print("\n🏆 Model Leaderboard:")
print(leaderboard)

# 🔹 Make predictions
future_forecasts = predictor.predict(train_data)
print("\n📈 Forecasted Values:")
print(future_forecasts.head())
