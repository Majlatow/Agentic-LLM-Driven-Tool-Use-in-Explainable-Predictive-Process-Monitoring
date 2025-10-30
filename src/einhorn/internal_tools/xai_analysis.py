"""
XAI (Explainable AI) analysis functions for manufacturing model interpretability.
Provides simplified access to SHAP-based explanations and model predictions.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import os
import sys

# Add the ML_XAI_UQ directory to the path to import the functions
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_xai_dir = os.path.join(current_dir, "ML_XAI_UQ")
if ml_xai_dir not in sys.path:
    sys.path.append(ml_xai_dir)

try:
    from ML_UQ_XAI_functions import ManufacturingModelExplainer
    XAI_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"XAI functions not available: {e}")
    XAI_AVAILABLE = False

warnings.filterwarnings('ignore')

# Global explainer instance
_explainer_instance = None


def _to_scalar(value: Any) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return 0.0
        return float(value[0])
    try:
        return float(value)
    except Exception:
        return 0.0

def _get_explainer() -> Optional[ManufacturingModelExplainer]:
    """
    Get or create the manufacturing model explainer instance.
    
    Returns:
        ManufacturingModelExplainer instance or None if not available
    """
    global _explainer_instance
    
    if not XAI_AVAILABLE:
        return None
        
    if _explainer_instance is None:
        try:
            # Use the correct paths for the models and data
            models_dir = os.path.join(ml_xai_dir, "models")
            data_dir = os.path.join(ml_xai_dir, "ml_data_splits")
            
            _explainer_instance = ManufacturingModelExplainer(
                models_dir=models_dir,
                data_dir=data_dir
            )
        except Exception as e:
            warnings.warn(f"Failed to initialize explainer: {e}")
            return None
    
    return _explainer_instance

def get_available_traces(limit: int = 20) -> Dict[str, Any]:
    """Retrieve a sample of traces that can be analysed with the XAI utilities.

    Args:
        limit: Maximum number of traces to return.

    Example:
        >>> response = get_available_traces(5)
        >>> response["success"], len(response.get("traces", []))
        (True, 5)

    User request prompt example:
        "List a few traces I can analyse with the XAI tools."

    Verbalization of outcome:
        "get_available_traces returns metadata about traces so the planner can pick
        valid identifiers for downstream analysis. Output dictionary contains
        'success', 'traces', and 'total_traces'."
    """
    explainer = _get_explainer()
    if explainer is None:
        return {
            "success": False,
            "error": "XAI functionality not available",
            "traces": pd.DataFrame()
        }
    
    try:
        traces = explainer.list_available_traces(limit)
        return {
            "success": True,
            "traces": traces.to_dict('records') if not traces.empty else [],
            "total_traces": len(traces)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traces": []
        }

def get_event_SHAP_explanation(
    trace_id: str,
    event_position: int = 0,
    show_plots: bool = False,
    save_plots: bool = True,
) -> Dict[str, Any]:
    """Generate a full SHAP explanation for a single manufacturing event.

    Args:
        trace_id: Trace identifier to analyse.
        event_position: Zero-based event index within the trace.
        show_plots: Display Matplotlib plots interactively when ``True``.
        save_plots: Persist generated plots when ``True``.

    Example:
        >>> get_event_SHAP_explanation("178864", 0, show_plots=False)
        {"success": True, "trace_id": "178864", ...}

    User request prompt example:
        "Explain why event 0 in trace 178864 takes so long."

    Verbalization of outcome:
        "get_event_SHAP_explanation returns a dictionary with event metadata,
        prediction intervals, raw SHAP contributions, and a ranked list of the
        top features. The structure enables agents to reference
        ['event_details', 'predictions', 'shap_values', 'top_features'] directly."
    """
    explainer = _get_explainer()
    if explainer is None:
        return {
            "success": False,
            "error": "XAI functionality not available"
        }

    try:
        # Validate inputs
        if not trace_id or not str(trace_id).strip():
            return {
                "success": False,
                "error": "trace_id must be a non-empty string"
            }

        if event_position < 0:
            return {
                "success": False,
                "error": "event_position must be non-negative"
            }

        # Perform the analysis
        results = explainer.analyze_event_shap(
            trace_id=str(trace_id),
            event_position=event_position,
            show_plots=show_plots,
            save_plots=save_plots
        )

        # Extract key information
        event_details = results.get('event_details', {})
        predictions = results.get('predictions', {})
        shap_values = results.get('shap_values', {})

        # Get top contributing features
        top_features = extract_SHAP_feature_importance(shap_values.get('main_model', {}), limit=5)

        return {
            "success": True,
            "trace_id": results.get('trace_id'),
            "event_position": results.get('event_position'),
            "event_details": {
                "activity": event_details.get('activity', 'Unknown'),
                "machine": event_details.get('machine_identifier', 'Unknown'),
                "actual_time": float(event_details.get('actual_time', 0.0))
            },
            "predictions": {
                "point_prediction": float(predictions.get('point_prediction', 0.0)),
                "lower_bound_90": float(predictions.get('lower_bound_90', 0.0)),
                "upper_bound_90": float(predictions.get('upper_bound_90', 0.0)),
                "interval_width": float(predictions.get('interval_width', 0.0))
            },
            "shap_values": {
                key: dict(values) if isinstance(values, dict) else values
                for key, values in shap_values.items()
            },
            "top_features": top_features,
            "prediction_error": abs(float(predictions.get('point_prediction', 0.0)) -
                                  float(event_details.get('actual_time', 0.0))),
            "within_interval": (float(predictions.get('lower_bound_90', 0.0)) <=
                              float(event_details.get('actual_time', 0.0)) <=
                              float(predictions.get('upper_bound_90', 0.0)))
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def get_event_processing_time_predictions(*trace_event_requests: Union[str, Tuple[str, Any]]) -> Dict[str, Any]:
    """Return prediction summaries for one or more trace events across traces.

    Args:
        *trace_event_requests: Each entry is a trace identifier optionally followed
            by event indices (single index or iterable). If no indices are supplied,
            all events in the trace are analysed.

    Example:
        >>> get_event_processing_time_predictions(("768451", 0), ("456687",))
        {"success": True, "events": [...]} 

    User request prompt example:
        "For traces 768451 and 456687 find the event with the longest predicted processing time."

    Verbalization of outcome:
        "get_event_processing_time_predictions returns a dictionary containing 'requested', 'events',
        and 'events_by_trace'. Each entry in 'events' stores the trace id, event
        position, status, and prediction interval fields so planners can identify
        the slowest event or compute statistics per trace."
    """
    explainer = _get_explainer()
    if explainer is None:
        return {
            "success": False,
            "error": "XAI functionality not available"
        }

    try:
        raw_results = explainer.analyze_events(*trace_event_requests)
        return {
            "success": True,
            "requested": raw_results.get("requested", []),
            "events": raw_results.get("events", []),
            "events_by_trace": raw_results.get("events_by_trace", {}),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc)
        }

def extract_SHAP_feature_importance(shap_values: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    """Summarise SHAP attributions into ranked feature importance entries.

    Args:
        shap_values: Mapping of feature names to SHAP scores (scalars or sequences).
        limit: Maximum number of features to return.

    Example:
        >>> extract_SHAP_feature_importance({"speed": 0.5, "temperature": -0.2})
        [{'rank': 1, 'feature': 'speed', ...}]

    User request prompt example:
        "List the top factors increasing processing time for this event."

    Verbalization of outcome:
        "extract_SHAP_feature_importance returns a list of dictionaries sorted by
        absolute impact. Each dictionary contains 'rank', 'feature', 'shap_value',
        'direction', and 'abs_importance' to support structured reporting."
    """
    if not shap_values:
        return []

    try:
        sorted_features = sorted(shap_values.items(), key=lambda x: abs(_to_scalar(x[1])), reverse=True)

        top_features: List[Dict[str, Any]] = []
        for i, (feature, shap_val) in enumerate(sorted_features[:limit]):
            shap_val_scalar = _to_scalar(shap_val)
            top_features.append(
                {
                    "rank": i + 1,
                    "feature": feature,
                    "shap_value": shap_val_scalar,
                    "direction": "increases" if shap_val_scalar > 0 else "decreases",
                    "abs_importance": abs(shap_val_scalar),
                }
            )

        return top_features

    except Exception as e:
        warnings.warn(f"Error extracting top features: {e}")
        return []

def validate_event_data(trace_id: str, event_position: int) -> Dict[str, Any]:
    """Validate event data before analysis.

    Args:
        trace_id: The TraceID to validate
        event_position: Position of the event to validate

    Example:
        >>> validate_event_data("178864", 0)
        {"valid": True, "trace_events_count": 42, ...}

    User request prompt example:
        "Check whether event 0 of trace 178864 is available for SHAP analysis."

    Verbalization of outcome:
        "validate_event_data returns a dictionary with 'valid', 'trace_events_count',
        and event metadata so planners can confirm whether downstream analyses can
        proceed."
    """
    explainer = _get_explainer()
    if explainer is None:
        return {
            "valid": False,
            "error": "XAI functionality not available"
        }
    
    try:
        # Check if trace exists
        trace_events = explainer.get_trace_events(trace_id)
        if len(trace_events) == 0:
            return {
                "valid": False,
                "error": f"No events found for TraceID '{trace_id}'"
            }
        
        # Check if event position is valid
        if event_position >= len(trace_events):
            return {
                "valid": False,
                "error": f"Event position {event_position} is out of range. Trace has {len(trace_events)} events."
            }
        
        # Get the specific event
        event = explainer.get_event_by_position(trace_id, event_position)
        
        return {
            "valid": True,
            "trace_events_count": len(trace_events),
            "event_position": event_position,
            "event_activity": str(event.get('activity', 'Unknown')),
            "event_machine": str(event.get('machine_identifier', 'Unknown'))
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }

def quick_event_analysis(trace_id: str, event_position: int = 0) -> Dict[str, Any]:
    """Perform a lightweight event analysis without interactive plotting.

    Example:
        >>> quick_event_analysis("178864", 0)
        {"success": True, ...}

    User request prompt example:
        "Quickly analyse event 0 for trace 178864 without generating plots."

    Verbalization of outcome:
        "quick_event_analysis returns a dictionary with 'success', 'trace_id',
        'event_position', and status flags so agents know if SHAP processing
        completed while suppressing plots."
    """
    if not XAI_AVAILABLE:
        return {
            "success": False,
            "error": "XAI functionality not available"
        }
    
    try:
        # Use the explainer instance with correct paths instead of quick_shap_analysis
        explainer = _get_explainer()
        if explainer is None:
            return {
                "success": False,
                "error": "Failed to initialize explainer"
            }
        
        # Perform the analysis directly
        results = explainer.analyze_event_shap(
            trace_id=str(trace_id),
            event_position=event_position,
            show_plots=False,
            save_plots=True
        )
        
        return {
            "success": True,
            "trace_id": results.get('trace_id'),
            "event_position": results.get('event_position'),
            "analysis_completed": True,
            "plots_saved": True
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        } 