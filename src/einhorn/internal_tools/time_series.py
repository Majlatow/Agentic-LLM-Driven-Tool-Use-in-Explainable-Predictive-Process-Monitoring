"""
Basic time series analysis functions.
"""
import numpy as np
from typing import List, Dict, Union, Tuple
from scipy import signal

def calculate_moving_average(data: List[float], window: int = 3) -> List[float]:
    """
    Calculate simple moving average of a time series.
    
    Args:
        data: Time series data
        window: Window size for moving average
        
    Returns:
        List of moving averages
    """
    if not data:
        raise ValueError("Data cannot be empty")
    if window < 1:
        raise ValueError("Window size must be positive")
    if window > len(data):
        raise ValueError("Window size cannot be larger than data length")
        
    np_data = np.array(data)
    weights = np.ones(window) / window
    ma = np.convolve(np_data, weights, mode='valid')
    return ma.tolist()

def detect_seasonality(data: List[float], max_period: int = None) -> Dict[str, Union[int, float]]:
    """
    Detect potential seasonality in time series using autocorrelation.
    
    Args:
        data: Time series data
        max_period: Maximum period to consider (default: len(data)//2)
        
    Returns:
        Dictionary containing detected period and correlation strength
    """
    if not data:
        raise ValueError("Data cannot be empty")
    if len(data) < 4:
        raise ValueError("Data length must be at least 4 points")
        
    np_data = np.array(data)
    if max_period is None:
        max_period = len(data) // 2
        
    # Calculate autocorrelation
    autocorr = np.correlate(np_data - np.mean(np_data), 
                          np_data - np.mean(np_data), 
                          mode='full')[len(np_data)-1:]
    
    # Normalize
    autocorr = autocorr / autocorr[0]
    
    # Find peaks
    peaks, _ = signal.find_peaks(autocorr[:max_period])
    
    if len(peaks) == 0:
        return {"period": 0, "strength": 0.0}
        
    # Get strongest peak
    peak_strengths = autocorr[peaks]
    strongest_peak = peaks[np.argmax(peak_strengths)]
    
    return {
        "period": int(strongest_peak),
        "strength": float(autocorr[strongest_peak])
    }

def calculate_trend(data: List[float]) -> Dict[str, float]:
    """
    Calculate linear trend in time series.
    
    Args:
        data: Time series data
        
    Returns:
        Dictionary containing slope and intercept of linear trend
    """
    if not data:
        raise ValueError("Data cannot be empty")
    if len(data) < 2:
        raise ValueError("Need at least 2 points to calculate trend")
        
    x = np.arange(len(data))
    y = np.array(data)
    
    # Calculate linear regression
    slope, intercept = np.polyfit(x, y, 1)
    
    return {
        "slope": float(slope),
        "intercept": float(intercept)
    } 