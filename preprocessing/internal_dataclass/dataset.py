from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class Station:
    """Represents a station with its position and all observations."""
    name: str
    position: List[float]  # [X, Y, Z]
    file_path: str

@dataclass
class GeoDataset:
    """Collection of stations with their observations."""
    samples: List[Station]