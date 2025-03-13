import os
import requests


start_num = 1
end_num = 999  

# Folder to save downloaded files
save_folder = "data_scrape_japan"
os.makedirs(save_folder, exist_ok=True)

# Base URL format
base_url = "https://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/{}.tenv3"

# Loop through J001 to J999
for i in range(start_num, end_num + 1):
    station_code = f"J{i:03d}"  # Format as J001, J002, ..., J999
    url = base_url.format(station_code)
    
    response = requests.get(url)

    if response.status_code == 200:  # Check if file exists
        file_path = os.path.join(save_folder, f"{station_code}.txt")
        
        with open(file_path, "wb") as file:
            file.write(response.content)
        
        print(f"Saved: {file_path}")
    else:
        print(f"File not found for {station_code}: HTTP {response.status_code}")
