"""
Internal tools package for statistical analysis and XAI.
"""
from .descriptive_stats import calculate_summary_stats, calculate_percentiles, detect_outliers
from .correlation_analysis import pearson_correlation, spearman_correlation, correlation_matrix
from .time_series import calculate_moving_average, detect_seasonality, calculate_trend
from .xai_analysis import (
    get_available_traces,
    get_event_SHAP_explanation,
    get_event_processing_time_predictions,
    extract_SHAP_feature_importance,
    validate_event_data,
    quick_event_analysis,
)

__all__ = [
    'calculate_summary_stats',
    'calculate_percentiles',
    'detect_outliers',
    'pearson_correlation',
    'spearman_correlation',
    'correlation_matrix',
    'calculate_moving_average',
    'detect_seasonality',
    'calculate_trend',
    'get_available_traces',
    'get_event_SHAP_explanation',
    'get_event_processing_time_predictions',
    'extract_SHAP_feature_importance',
    'validate_event_data',
    'quick_event_analysis',
] 