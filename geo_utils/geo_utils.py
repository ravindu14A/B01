import numpy as np
a = 6378137
b = 6356752.314245
e2 = (a**2 - b**2) / a**2

def xyz_to_geodetic(x,y,z):
	lon = np.arctan2(y,x)

	e2 = (a**2 - b**2) / a**2  # eccentricity squared

	r = np.sqrt(x**2 + y**2)

	lat = np.arctan(z / (r * (1 - e2)))

	# Step 3: Iterative computation
	tolerance = 1e-12  # Convergence threshold
	delta = 1  # Initial delta for the loop
	while delta > tolerance:
		N = a / np.sqrt(1 - e2 * np.sin(lat)**2)  # Radius of curvature
		h = r / np.cos(lat) - N  # Height above ellipsoid
		lat_new = np.arctan(z / (r * (1 - e2 * N / (N + h))))  # New latitude
		delta = np.abs(lat_new - lat)  # Check for convergence
		lat = lat_new  # Update latitude

	lat, lon = np.degrees(lat), np.degrees(lon)

	return lat, lon, h

def geodetic_to_xyz(lat, lon, h):
	# Eccentricity squared

	# Radius of curvature in the prime vertical
	N = a / np.sqrt(1 - e2 * np.sin(lat)**2)

	# ECEF coordinates
	x = (N + h) * np.cos(lat) * np.cos(lon)
	y = (N + h) * np.cos(lat) * np.sin(lon)
	z = ((1 - e2) * N + h) * np.sin(lat)

	return x, y, z

def radius_of_curvature(lat, direction='north'):
    """Calculate radius of curvature for north or east direction at given latitude"""
    sin_lat = np.sin(lat)
    if direction == 'north':
        return a * (1 - e2) / (1 - e2 * sin_lat**2)**(1.5)
    elif direction == 'east':
        return a / np.sqrt(1 - e2 * sin_lat**2)
    else:
        raise ValueError("direction must be 'north' or 'east'")

def displacement(lat1, lon1, lat2, lon2, xyz1=None, xyz2=None):
    """
    Calculate north/east displacement between two points.
    
    Parameters:
    - lat1, lon1: Latitude/longitude of first point (degrees)
    - lat2, lon2: Latitude/longitude of second point (degrees)
    - xyz1: Optional XYZ coordinates of first point (meters)
    - xyz2: Optional XYZ coordinates of second point (meters)
    
    Returns:
    - (d_north, d_east) tuple in millimeters
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    
    if xyz1 is not None and xyz2 is not None:
        # Use XYZ coordinates for more accurate calculation
        x1, y1, z1 = xyz1
        x2, y2, z2 = xyz2
        
        # Calculate vector between points in ECEF
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        
        # Rotate to local ENU (East-North-Up) frame at point1
        sin_lat = np.sin(lat1_rad)
        cos_lat = np.cos(lat1_rad)
        sin_lon = np.sin(lon1_rad)
        cos_lon = np.cos(lon1_rad)
        
        # ENU transformation matrix
        e = -sin_lon*dx + cos_lon*dy
        n = -sin_lat*cos_lon*dx - sin_lat*sin_lon*dy + cos_lat*dz
        u = cos_lat*cos_lon*dx + cos_lat*sin_lon*dy + sin_lat*dz
        
        # Convert to millimeters
        return n*1000, e*1000
    else:
        # Fall back to approximate method using radius of curvature
        M = radius_of_curvature(lat1_rad, direction='north')  # meridian radius
        N = radius_of_curvature(lat1_rad, direction='east')   # prime vertical radius
        
        d_north = M * delta_lat
        d_east = N * np.cos(lat1_rad) * delta_lon
        
        return d_north*1000, d_east*1000



# to do implement func for error(cringe)


#print(xyz_to_geodetic(-1131051.99836813, 6236311.69761652, 711748.157607914))