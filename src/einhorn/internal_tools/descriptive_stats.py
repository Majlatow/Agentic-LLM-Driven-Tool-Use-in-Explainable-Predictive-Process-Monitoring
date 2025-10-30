"""
Basic descriptive statistics functions for numerical data analysis.
"""
import numpy as np
from typing import List, Dict, Union

def calculate_summary_stats(data: List[float]) -> Dict[str, float]:
    """
    Calculate basic summary statistics for a list of numbers.
    
    Args:
        data: List of numerical values
        
    Returns:
        Dictionary containing mean, median, std_dev, min, max
    """
    if not data:
        raise ValueError("Data list cannot be empty")
        
    np_data = np.array(data)
    return {
        "mean": float(np.mean(np_data)),
        "median": float(np.median(np_data)),
        "std_dev": float(np.std(np_data)),
        "min": float(np.min(np_data)),
        "max": float(np.max(np_data))
    }

def calculate_percentiles(data: List[float], percentiles: List[float] = [25, 50, 75]) -> Dict[str, float]:
    """
    Calculate specified percentiles for a list of numbers.
    
    Args:
        data: List of numerical values
        percentiles: List of percentiles to calculate (default: [25, 50, 75])
        
    Returns:
        Dictionary mapping percentile values to their computed values
    """
    if not data:
        raise ValueError("Data list cannot be empty")
    if not all(0 <= p <= 100 for p in percentiles):
        raise ValueError("Percentiles must be between 0 and 100")
        
    np_data = np.array(data)
    return {
        f"p{int(p)}": float(np.percentile(np_data, p))
        for p in percentiles
    }

def detect_outliers(data: List[float], threshold: float = 1.5) -> List[bool]:
    """
    Detect outliers using the IQR method.
    
    Args:
        data: List of numerical values
        threshold: IQR multiplier for outlier detection (default: 1.5)
        
    Returns:
        List of boolean values indicating outlier status for each data point
    """
    if not data:
        raise ValueError("Data list cannot be empty")
        
    np_data = np.array(data)
    q1 = np.percentile(np_data, 25)
    q3 = np.percentile(np_data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    return [(x < lower_bound) or (x > upper_bound) for x in data] 