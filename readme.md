# Predicting the Next Magnitude 9+ Earthquake at the Sumatra-Andaman Subduction Interface


This project analyzes GNSS time-series data from Malaysian and Thai stations to forecast when the next magnitude 9+ earthquake could occur at the Sumatra-Andaman subduction interface. 

## Key Findings

- **Earthquake Prediction**: Next large earthquake predicted between **2208-2228** (95% confidence interval)
- **Station-Specific Predictions**:
  - ARAU (Malaysia): 2331 ± 24.2 years
  - GETI (Malaysia): 2190 ± 15.2 years
  - KUAL (Malaysia): 2382 ± 53.6 years
  - USMP (Malaysia): 2182 ± 6.6 years
  - PHUK (Thailand): 2290 ± 10.1 years

## Project Structure

```
B01/
├── data/
│   ├── raw/                    # Raw GNSS data files
│   ├── processed/              # Cleaned and processed data
│   └── partially_processed_steps/ # Data at different stages of preprocessing 
├── preprocessing/
│   ├── convert.py              # Data format conversion
│   ├── coordinate.py           # Coordinate system transformations
│   ├── data_transfer.py        # Data handling utilities
│   ├── delete_earthquakes.py   # Earthquake discontinuity removal
│   ├── pca.py                  # Principal Component Analysis
│   └── plate_motion.py         # Plate motion calculations
├── predictions/
│   ├── main.py                 # Main prediction script
│   └── predictions.py          # Core prediction algorithms
└── README.md
```


## Installation

1. Clone the repository:
```bash
    git clone [repository-url]
    cd B01
```

2. Install required dependencies:
```bash
    pip install -r requirements.txt
```

3. Ensure you have the GNSS data files in the `data/raw/` directory. To allow for easier usage, there are additional 
folders with the data after each stage of preprocessing. If one wants to directly generate predictions, the files 
in the `data/processed/` directory, already include all preprocessing. 

### Running the Analysis

1. **Data Preprocessing**:
```bash
    python preprocessing/convert.py          # Convert data formats
    python preprocessing/coordinate.py       # Transform coordinates to NEU
    python preprocessing/delete_earthquakes.py  # Remove discontinuities
    python preprocessing/plate_motion.py     # Remove absolute plate motion
    python preprocessing/pca.py              # Apply PCA transformation
```

2. **Generate Predictions**:
```bash
    python predictions/main.py
```

## Methodology

### Data Processing Pipeline

1. **Coordinate Transformation**: Convert ECEF coordinates to North-East-Up (NEU) local frame
2. **Discontinuity Detection**: Identify and remove earthquake-caused jumps from the time series
3. **Plate Motion Removal**: Subtract absolute Sunda plate movement
4. **Principal Component Analysis**: Reduce dimensionality and focus on main deformation direction
5. **Regression Modeling**: Fit post-seismic deformation using exponential decay model

### Mathematical Model

The post-seismic deformation is modeled using:

```
f(t) = vt + c₁e^(-u₁t) + c₂e^(-u₂t) + d
```

Where:
- `v`: velocity of station before earthquake
- `c₁, c₂`: exponential decay coefficients
- `u₁, u₂`: decay rates
- `d`: constant offset

### Error Estimation

Monte Carlo simulation with 1000 iterations to calculate prediction uncertainty:
- Generate synthetic datasets within measurement uncertainty (From the provided covariance)
- Determine 95% confidence intervals

## Key Features

- **Automated Data Processing**: Complete pipeline from raw GNSS data to predictions
- **Error Handling**: Uncertainty quantification
- **Visualization Tools**: Generate prediction plots and uncertainty bounds

## Study Area

The analysis focuses on:
- **Primary Region**: Sumatra-Andaman subduction interface
- **Countries**: Thailand and Malaysia
- **Stations**: 7 long-duration GNSS stations (1999-2024)
- **Key Cities**: Bangkok, Phuket, Kuala Lumpur, Sabah

## Limitations & Considerations

- **Data Length**: Limited to ~25 years of observations
- **Seismic Cycle**: Analysis based on single major earthquake (2004)
- **Model Assumptions**: Assumes repeating seismic cycle behavior

## Usage Examples

### Basic Prediction Run

```python
from predictions.predictions import monte
from preprocessing.coordinate import coordinate_transform

# Set parameters
country = "thailand"
station = "PHUK"
N = 50  # Monte Carlo samples
years_predict = 300
confidence_level = 95

# Run prediction
monte(country, station, N, years_predict, confidence_level, offset=60, pred_pos=4)
```

## Scientific Background

This research addresses the challenge of earthquake prediction in one of the world's most seismically active regions. The 2004 Sumatra-Andaman earthquake (Mw 9.1) killed over 225,000 people, highlighting the critical need for improved forecasting methods.

The approach leverages the concept of seismic cycles - repeating patterns of stress accumulation and release along tectonic faults. By analyzing ground deformation patterns captured by GNSS stations, the model attempts to identify when conditions similar to the 2004 earthquake might recur.

## Contributing

This project was developed as part of the Test, Analysis and Simulation course (AE2224-1) at TU Delft's Aerospace Faculty. 

**Team Members**:
- Ravindu Kalinga Bandara
- Nikolai Da Silva
- Laura Kaczmarzyk
- Aleksandra Pawlik
- Andrei Potfalean
- Maxim Shcherbina
- Lowie Thierens
- Nicolas Vardon
- Viktor Velichkov
- Toine van Wassenaar
