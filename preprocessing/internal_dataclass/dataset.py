from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd

@dataclass
class Station:
    """Represents a station with its position and all observations."""
    name: str
    position: List[float]  # [X, Y, Z]
    data: pd.DataFrame

@dataclass
class GeoDataset:
    """Collection of stations with their observations."""
    samples: List[Station]