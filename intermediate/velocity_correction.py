import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd

def plate_motion_api(lat, lon, h):#this is not too slow for only all stations
	url = "https://www.unavco.org/software/geodetic-utilities/plate-motion-calculator/plate-motion/model"
	d_lat = int(lat)
	m_lat = int((lat - d_lat) * 60)
	s_lat = (lat - d_lat - m_lat/60) * 3600

	d_lon = int(lon)
	m_lon = int((lon - d_lon) * 60)
	s_lon = (lon - d_lon - m_lon/60) * 3600

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
		return north_velocity, east_velocity
	else:
		raise ValueError("Could not find enough velocity data in the page.")



def euler_vector(pole_lat, pole_lon, omega_deg_per_myr, clockwise=True):
	# Convert omega to radians per year
	omega = omega_deg_per_myr * (np.pi / 180) / 1e6  # rad/year
	if clockwise:
		omega *= -1
	pole_lat_rad = np.radians(pole_lat)
	pole_lon_rad = np.radians(pole_lon)
	
	wx = omega * np.cos(pole_lat_rad) * np.cos(pole_lon_rad)
	wy = omega * np.cos(pole_lat_rad) * np.sin(pole_lon_rad)
	wz = omega * np.sin(pole_lat_rad)
	return np.array([wx, wy, wz])

def plate_motion(x,y,z,lat,lon):
	#print(lat, lon)
	pole_lat = 49.0
	pole_lon = -94.2
	omega_deg_per_myr = 0.336
	clockwise=True
	w = euler_vector(pole_lat, pole_lon, omega_deg_per_myr, clockwise)
	lon_rad = np.radians(lon)
	lat_rad = np.radians(lat)

	v_xyz = np.cross(w, np.array([x,y,z]))

	R = np.array([
		[-np.sin(lon_rad),              np.cos(lon_rad),               0],
		[-np.sin(lat_rad)*np.cos(lon_rad), -np.sin(lat_rad)*np.sin(lon_rad), np.cos(lat_rad)],
		[np.cos(lat_rad)*np.cos(lon_rad),  np.cos(lat_rad)*np.sin(lon_rad),  np.sin(lat_rad)]
	])
	v_relative = -R @ v_xyz # not sure why this is negative
	ve,vn = v_relative[0], v_relative[1] # not sure why e,n flipped
	return vn*1000, ve*1000


#print(plate_motion(-2.583275e+06 , 5.811931e+06 , -4.772896e+05 ,  -4.320442 , 113.964090))

#input lat, lon, h (deg,deg,meters)
#return v_n,v_e in mm/day

def create_velocity_df_vectorized(df):
	"""
	Vectorized version of velocity DataFrame creation.
	"""
	# Get first position per station
	first_positions = df.groupby('station').first().reset_index()
	
	# Calculate velocities
	velocities = first_positions.apply(
		lambda row: pd.Series(plate_motion(row['x'], row['y'], row['z'], row['lat'], row['lon'])),
		axis=1
	)
	velocities.columns = ['vn', 've']
	#print(first_positions[['x','y','z',"lat",'lon']])
	
	# Combine with station names
	return pd.concat([first_positions['station'],velocities], axis=1)

def apply_velocity_correction(df):
	velocity_df = create_velocity_df_vectorized(df)
	#print(velocity_df)
	# Create dictionary of velocities for quick lookup
	velocity_dict = velocity_df.set_index('station').to_dict('index')
	
	# Group by station and process each group separately
	def process_group(group):
		station = group.name
		ref_date = group['date'].iloc[0]
		vn = velocity_dict[station]['vn']
		ve = velocity_dict[station]['ve']

		print(vn, ve, station)
		
		group['delta_years'] = (group['date'] - ref_date).dt.total_seconds() / (365.25 * 24 * 3600)
		group['d_north_mm'] = group['d_north_mm'] - (vn * group['delta_years'])
		group['d_east_mm'] = group['d_east_mm'] - (ve * group['delta_years'])
		return group.drop(columns=['delta_years'])
	
	return df.groupby('station', group_keys=False).apply(process_group)
