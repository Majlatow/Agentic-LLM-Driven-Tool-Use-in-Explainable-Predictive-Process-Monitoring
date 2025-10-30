"""
External tools imported from various Python packages.
"""
from typing import List, Dict, Union, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from geopy.distance import geodesic

def kmeans_clustering(data: List[List[float]], n_clusters: int = 3) -> Dict[str, Union[List[int], List[List[float]]]]:
    """
    Perform K-means clustering on the input data.
    
    Args:
        data: List of data points (each point is a list of features)
        n_clusters: Number of clusters to form
        
    Returns:
        Dictionary containing cluster labels and centroids
    """
    if not data or not data[0]:
        raise ValueError("Data cannot be empty")
    if n_clusters < 1:
        raise ValueError("Number of clusters must be positive")
    if n_clusters > len(data):
        raise ValueError("Number of clusters cannot exceed number of data points")
        
    np_data = np.array(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(np_data)
    
    return {
        "labels": labels.tolist(),
        "centroids": kmeans.cluster_centers_.tolist()
    }

def linear_regression_predict(X_train: List[List[float]], y_train: List[float], 
                            X_predict: List[List[float]]) -> Dict[str, Union[List[float], List[float], float]]:
    """
    Train a linear regression model and make predictions.
    
    Args:
        X_train: Training features
        y_train: Training target values
        X_predict: Features to predict
        
    Returns:
        Dictionary containing predictions, coefficients, and R-squared score
    """
    if not X_train or not y_train or not X_predict:
        raise ValueError("Input data cannot be empty")
    if len(X_train) != len(y_train):
        raise ValueError("Number of training examples must match number of target values")
    if len(X_train[0]) != len(X_predict[0]):
        raise ValueError("Training and prediction features must have same dimensionality")
        
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_predict)
    r2_score = model.score(X_train, y_train)
    
    return {
        "predictions": predictions.tolist(),
        "coefficients": model.coef_.tolist(),
        "r2_score": float(r2_score)
    }

def haversine_distance(points: List[Tuple[float, float]]) -> Dict[str, Union[float, List[float]]]:
    """
    Calculate distances between consecutive pairs of lat-long coordinates using the Haversine formula.
    
    Args:
        points: List of (latitude, longitude) tuples
        
    Returns:
        Dictionary containing total distance and segment distances in kilometers
    """
    if len(points) < 2:
        raise ValueError("Need at least 2 points to calculate distances")
        
    segment_distances = []
    total_distance = 0.0
    
    for i in range(len(points) - 1):
        point1 = points[i]
        point2 = points[i + 1]
        
        distance = geodesic(point1, point2).kilometers
        segment_distances.append(distance)
        total_distance += distance
    
    return {
        "total_distance": float(total_distance),
        "segment_distances": segment_distances
    } 