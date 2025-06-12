import os
import pickle
import pandas as pd
from datetime import timedelta
import numpy as np
# import main as mn
import requests
from bs4 import BeautifulSoup
# Load data
country = "thailand"


def plate_motion_api(lat, lon, h):  # this is not too slow for only all stations
    url = "https://www.unavco.org/software/geodetic-utilities/plate-motion-calculator/plate-motion/model"
    d_lat = int(lat)
    m_lat = int((lat - d_lat) * 60)
    s_lat = (lat - d_lat - m_lat / 60) * 3600

    d_lon = int(lon)
    m_lon = int((lon - d_lon) * 60)
    s_lon = (lon - d_lon - m_lon / 60) * 3600

    params = {
        'lat': d_lat,
        'lat_m': m_lat,
        'lat_s': s_lat,
        'lon': d_lon,
        'lon_m': m_lon,
        'lon_s': s_lon,
        'h': h,
        'x': '',
        'y': '',
        'z': '',
        'site': '',
        'geo': '',
        'xyz': '',
        'model': 'itrf2014',
        'plate': 'usr-p',
        'reference': 'NNR',
        'up_lat': 49,
        'up_lon': -94.2,
        'up_w': .34,
        'up_x': '',
        'up_y': '',
        'up_z': '',
        'ur_lat': '',
        'ur_lon': '',
        'ur_w': '',
        'ur_x': '',
        'ur_y': '',
        'ur_z': '',
        'format': 'html',
        'submit': 'Submit'
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    right_aligned_tds = soup.find_all('td', {'align': 'right'})

    if len(right_aligned_tds) >= 2:
        north_velocity = float(right_aligned_tds[0].text.strip())
        east_velocity = float(right_aligned_tds[1].text.strip())
        return north_velocity/10, east_velocity/10
    else:
        raise ValueError("Could not find enough velocity data in the page.")




directory = f"../data/partially_processed_steps/{country}/filtered_cm"
directory1 = f"../data/partially_processed_steps/{country}/raw_pickle"

directory_out = f"../data/partially_processed_steps/{country}/filtered_cm_normalised"

for filename in os.listdir(directory_out):
    file_path = os.path.join(directory_out, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

for filename in os.listdir(directory):
    if filename.endswith('.pkl'):
        filepath = os.path.join(directory, filename)
        filepath1 = os.path.join(directory1, filename)

        with open(filepath1, 'rb') as f:

            data1 = pickle.load(f)
            df1 = pd.DataFrame(data1)

            lat = df1["lat"][0]
            long = df1["long"][0]
            alt = df1["alt"][0]

            North, East = plate_motion_api(np.degrees(lat), np.degrees(long), alt)
            if filename == "PHUK.pkl":
                North = -0.55  # cm/year
                East = 2.9  # cm/year

            print(f"station = {filename}, North = {North}, East = {East}")

            North = North / 365.25
            East = East / 365.25

        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            df = pd.DataFrame(data)

            begin = df["date"].iloc[0]
            days = (df["date"] - begin) / pd.Timedelta(days=1)

            df["lat"] = df["lat"] - North * days
            df["long"] = df["long"] - East * days

    filename = os.path.join(directory_out, f"{filename}")
    df.to_pickle(filename)