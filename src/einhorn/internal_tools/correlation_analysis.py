"""
Correlation analysis functions for examining relationships between variables.
"""
import numpy as np
from typing import List, Dict, Union, Tuple
from scipy import stats

def pearson_correlation(x: List[float], y: List[float]) -> Dict[str, float]:
    """
    Calculate Pearson correlation coefficient and p-value between two variables.
    
    Args:
        x: First variable data
        y: Second variable data
        
    Returns:
        Dictionary containing correlation coefficient and p-value
    """
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length")
    if not x or not y:
        raise ValueError("Input arrays cannot be empty")
        
    coef, p_value = stats.pearsonr(x, y)
    return {
        "correlation": float(coef),
        "p_value": float(p_value)
    }

def spearman_correlation(x: List[float], y: List[float]) -> Dict[str, float]:
    """
    Calculate Spearman rank correlation coefficient between two variables.
    
    Args:
        x: First variable data
        y: Second variable data
        
    Returns:
        Dictionary containing correlation coefficient and p-value
    """
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length")
    if not x or not y:
        raise ValueError("Input arrays cannot be empty")
        
    coef, p_value = stats.spearmanr(x, y)
    return {
        "correlation": float(coef),
        "p_value": float(p_value)
    }

def correlation_matrix(data: List[List[float]], method: str = 'pearson') -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate correlation matrix for multiple variables.
    
    Args:
        data: List of lists containing variables
        method: Correlation method ('pearson' or 'spearman')
        
    Returns:
        Tuple of (correlation matrix, p-value matrix)
    """
    if not data or not data[0]:
        raise ValueError("Data cannot be empty")
    if not all(len(row) == len(data[0]) for row in data):
        raise ValueError("All variables must have the same length")
        
    np_data = np.array(data)
    n_vars = len(data)
    corr_matrix = np.zeros((n_vars, n_vars))
    p_matrix = np.zeros((n_vars, n_vars))
    
    for i in range(n_vars):
        for j in range(i, n_vars):
            if method == 'pearson':
                corr, p_val = stats.pearsonr(np_data[i], np_data[j])
            else:
                corr, p_val = stats.spearmanr(np_data[i], np_data[j])
            
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
            p_matrix[i, j] = p_val
            p_matrix[j, i] = p_val
            
    return corr_matrix, p_matrix 