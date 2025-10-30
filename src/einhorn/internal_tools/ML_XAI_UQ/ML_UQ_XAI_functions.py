"""
ML_UQ_XAI_functions.py

Explainable AI (XAI) Functions for Manufacturing Model Analysis
Provides functionality for manual examination of the trained XGBoost model and its predictions.
Focuses on SHAP (SHapley Additive exPlanations) analysis for local interpretability.

Author: Generated for BDE Manufacturing Analysis Pipeline
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class ManufacturingModelExplainer:
    """
    A comprehensive class for explainable AI analysis of manufacturing prediction models.
    Provides SHAP-based local explanations for individual events and prediction intervals.
    """
    
    def __init__(self, models_dir: str = "bde_analysis_output/final_ML_UQ_results/models",
                 data_dir: str = "bde_analysis_output/ml_data_splits"):
        """Initialise the explainer with trained artefacts and data splits.

        Args:
            models_dir (str): Directory that stores the persisted XGBoost models
                and label encoders.
            data_dir (str): Directory that stores the train/validation/test CSV
                files and ``column_definitions.csv`` metadata.

        Returns:
            None

        Example:
            >>> explainer = ManufacturingModelExplainer(
            ...     models_dir="/models",
            ...     data_dir="/data_splits"
            ... )

        User request prompt example:
            "Initialize the manufacturing explainer so we can query predictions and SHAP results."

        Verbalization of outcome:
            "ManufacturingModelExplainer loads the trained models, metadata, and SHAP explainers so downstream tools can answer analysis requests."
        """
        self.models_dir = models_dir
        self.data_dir = data_dir
        
        # Initialize attributes
        self.main_model = None
        self.quantile_model_005 = None
        self.quantile_model_095 = None
        self.label_encoders = None
        self.test_data = None
        self.predictor_columns = None
        self.target_column = None
        self.additional_columns = None
        
        # SHAP explainers
        self.main_explainer = None
        self.quantile_explainer_005 = None
        self.quantile_explainer_095 = None
        
        # Load everything
        self._load_models()
        self._load_data()
        self._prepare_shap_explainers()
        
    def _load_models(self):
        """Load all trained models and encoders."""
        print("Loading trained models...")
        
        # Load main XGBoost model
        self.main_model = joblib.load(f"{self.models_dir}/final_xgboost_model.joblib")
        
        # Load quantile models for prediction intervals
        self.quantile_model_005 = joblib.load(f"{self.models_dir}/quantile_model_0.050.joblib")
        self.quantile_model_095 = joblib.load(f"{self.models_dir}/quantile_model_0.950.joblib")
        
        # Load label encoders
        self.label_encoders = joblib.load(f"{self.models_dir}/final_label_encoders.joblib")
        
        print("✓ Models loaded successfully")
        
    def _load_data(self):
        """Load test data and column definitions."""
        print("Loading test data...")
        
        # Load column definitions
        col_defs = pd.read_csv(f"{self.data_dir}/column_definitions.csv")
        self.predictor_columns = [col for col in col_defs['predictor_columns'].dropna()]
        self.target_column = col_defs['target_column'].dropna().iloc[0]
        self.additional_columns = [col for col in col_defs['additional_columns'].dropna()]
        
        # Load test data
        self.test_data = pd.read_csv(f"{self.data_dir}/bde_test_dataset.csv")
        
        print(f"✓ Test data loaded: {len(self.test_data):,} events")
        print(f"✓ Features: {len(self.predictor_columns)} predictor columns")
        
    def _prepare_shap_explainers(self):
        """Prepare SHAP explainers for all models."""
        print("Preparing SHAP explainers (this may take a moment)...")
        
        # Get a sample of test data for SHAP background
        sample_size = min(1000, len(self.test_data))
        background_data = self.test_data.sample(n=sample_size, random_state=42)[self.predictor_columns]
        
        # Preprocess background data
        background_processed = self._preprocess_features(background_data)
        
        # Create SHAP explainers
        self.main_explainer = shap.TreeExplainer(self.main_model, background_processed)
        self.quantile_explainer_005 = shap.TreeExplainer(self.quantile_model_005, background_processed)
        self.quantile_explainer_095 = shap.TreeExplainer(self.quantile_model_095, background_processed)
        
        print("✓ SHAP explainers ready")
        
    def _preprocess_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess features using the same preprocessing as during training.
        
        Args:
            features_df: DataFrame with raw features
            
        Returns:
            Preprocessed feature array
        """
        processed_features = features_df.copy()
        
        # Handle categorical columns with label encoding
        categorical_columns = ['activity', 'machine_identifier', 'product_type', 'material_extracted', 
                             'standard_extracted', 'work_type', 'previous_activity', 'previous_machine', 
                             'previous_activity_2', 'previous_machine_2', 'following_activity', 'following_machine']
        
        for col in categorical_columns:
            if col in processed_features.columns and col in self.label_encoders:
                # Convert to string and handle unknown values
                processed_features[col] = processed_features[col].astype(str)
                encoder = self.label_encoders[col]
                
                # Handle unknown values by assigning them to the most frequent class
                unknown_mask = ~processed_features[col].isin(encoder.classes_)
                if unknown_mask.any():
                    most_frequent_class = encoder.classes_[0]  # Assume first class is most frequent
                    processed_features.loc[unknown_mask, col] = most_frequent_class
                
                processed_features[col] = encoder.transform(processed_features[col])
        
        # Select only predictor columns and ensure they're in the right order
        final_features = pd.DataFrame()
        for col in self.predictor_columns:
            if col in processed_features.columns:
                final_features[col] = processed_features[col]
            else:
                final_features[col] = 0  # Fill missing columns with 0
        
        # Convert all columns to numeric, coercing errors to 0
        for col in final_features.columns:
            final_features[col] = pd.to_numeric(final_features[col], errors='coerce').fillna(0)
        
        # Convert to numpy array with float64 dtype
        return final_features.values.astype(np.float64)
        
    def get_trace_events(self, trace_id: str) -> pd.DataFrame:
        """Retrieve all recorded events for a given trace identifier.

        Args:
            trace_id (str): Identifier of the trace to inspect.

        Returns:
            pandas.DataFrame: Subset of the test dataset containing the
            requested trace’s events; empty when the trace is unknown.

        Example:
            >>> explainer = ManufacturingModelExplainer()
            >>> events = explainer.get_trace_events("768451")
            >>> len(events)
            12

        User request prompt example:
            "Show me all recorded events for trace 768451."

        Verbalization of outcome:
            "get_trace_events returns a DataFrame containing every event row for trace 768451, ready for downstream filtering or scoring."
        """
        # Convert trace_id to int to match the data type in the dataset
        try:
            trace_id_int = int(trace_id)
        except ValueError:
            print(f"Warning: Could not convert TraceID '{trace_id}' to integer")
            return pd.DataFrame()
            
        trace_events = self.test_data[self.test_data['TraceID'] == trace_id_int].copy()
        
        if len(trace_events) == 0:
            print(f"Warning: No events found for TraceID '{trace_id}' in test data")
            return pd.DataFrame()
            
        # Sort by event position if available
        if 'trace_position' in trace_events.columns:
            trace_events = trace_events.sort_values('trace_position')
        
        return trace_events
        
    def get_event_by_position(self, trace_id: str, event_position: int) -> pd.Series:
        """Fetch one event instance by its zero-based index within a trace.

        Args:
            trace_id (str): Identifier of the trace that contains the event.
            event_position (int): Zero-based index within the trace sequence.

        Returns:
            pandas.Series: Single-row series describing the event.

        Example:
            >>> explainer = ManufacturingModelExplainer()
            >>> event = explainer.get_event_by_position("768451", 0)
            >>> event["TraceID"]
            768451

        User request prompt example:
            "Fetch the first event for trace 768451 so I can inspect its features."

        Verbalization of outcome:
            "get_event_by_position returns a Series describing the requested event, including predictor columns and metadata such as activity and machine."
        """
        trace_events = self.get_trace_events(trace_id)
        
        if len(trace_events) == 0:
            raise ValueError(f"No events found for TraceID '{trace_id}'")
            
        if event_position >= len(trace_events):
            raise ValueError(f"Event position {event_position} out of range. Trace has {len(trace_events)} events.")
            
        return trace_events.iloc[event_position]

    def analyze_events(self, *trace_event_requests: Tuple[str, Optional[Any]]) -> Dict[str, Any]:
        """Compute point_prediction, lower_bound, upper_bound and interval_width for one or more trace events from one or more traces.

        Args:
            trace_event_requests (Tuple[str, Optional[Any]]): Each entry describes
                a trace and optional event indices. Provide a string trace identifier
                optionally followed by either a single zero-based index or an iterable
                of indices.

        Returns:
            Dict[str, Any]: Response with request metadata plus per-event prediction
            details (point estimate, 90% interval bounds, interval width).

        Example:
            >>> explainer = ManufacturingModelExplainer()
            >>> payload = explainer.analyze_events(("768451",), ("456687",))
            >>> payload["events"][0]["predictions"]["point"]
            123.45

        User request prompt example:
            "For the traces 768451 and 456687 find the event with the longest predicted processing time."

        Verbalization of outcome:
            "analyze_events yields a list of all events from the requested traces with their predicted processing times and interval information, enabling identification of the event position with the longest predicted duration."
        """
        if not trace_event_requests:
            raise ValueError("At least one trace specification must be provided.")

        normalized_requests: List[Dict[str, Any]] = []
        events_output: List[Dict[str, Any]] = []

        for request in trace_event_requests:
            if not request:
                continue

            if not isinstance(request, (tuple, list)):
                trace_id = str(request)
                positions_spec = None
            else:
                if len(request) == 0:
                    continue
                trace_id = str(request[0])
                positions_spec = request[1] if len(request) > 1 else None

            trace_events = self.get_trace_events(trace_id)
            if trace_events.empty:
                normalized_requests.append({
                    "trace_id": trace_id,
                    "event_positions": [],
                    "processed": False,
                    "reason": "No events found"
                })
                continue

            if positions_spec is None:
                event_indices = list(range(len(trace_events)))
            else:
                if isinstance(positions_spec, (list, tuple, set)):
                    event_indices = [int(pos) for pos in positions_spec]
                else:
                    event_indices = [int(positions_spec)]

            normalized_requests.append({
                "trace_id": trace_id,
                "event_positions": event_indices,
                "processed": True
            })

            for event_index in event_indices:
                if event_index < 0 or event_index >= len(trace_events):
                    events_output.append({
                        "trace_id": trace_id,
                        "event_position": int(event_index),
                        "status": "error",
                        "message": f"Event position {event_index} out of range"
                    })
                    continue

                event_series = trace_events.iloc[event_index]
                features = event_series[self.predictor_columns].to_frame().T
                processed_features = self._preprocess_features(features)

                point_prediction = float(self.main_model.predict(processed_features)[0])
                lower_bound = float(self.quantile_model_005.predict(processed_features)[0])
                upper_bound = float(self.quantile_model_095.predict(processed_features)[0])

                trace_position = event_series.get('trace_position')

                events_output.append({
                    "trace_id": trace_id,
                    "event_position": int(event_index),
                    "trace_position": int(trace_position) if pd.notna(trace_position) else None,
                    "predictions": {
                        "point": point_prediction,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "interval_width": upper_bound - lower_bound
                    },
                    "status": "ok"
                })

        grouped_events: Dict[str, List[Dict[str, Any]]] = {}
        for item in events_output:
            grouped_events.setdefault(item["trace_id"], []).append(item)

        return {
            "requested": normalized_requests,
            "events": events_output,
            "events_by_trace": grouped_events
        }
        
    def analyze_event_shap(self, trace_id: str, event_position: int, 
                          show_plots: bool = True, save_plots: bool = True,
                          output_dir: str = "bde_analysis_output/xai_analysis") -> Dict[str, Any]:
        """Produce SHAP explanations and prediction summary for one event.

        Args:
            trace_id (str): Identifier of the trace that contains the event.
            event_position (int): Zero-based index of the event within the trace.
            show_plots (bool): Display Matplotlib plots interactively when ``True``.
            save_plots (bool): Persist generated plots to ``output_dir`` when ``True``.
            output_dir (str): Target directory for exported visualisations.

        Returns:
            Dict[str, Any]: Structured bundle with event metadata, prediction
            intervals and SHAP values per feature.

        Example:
            >>> explainer = ManufacturingModelExplainer()
            >>> details = explainer.analyze_event_shap("768451", 0, show_plots=False)
            >>> details["event_details"]["activity"]
            'Milling'

        User request prompt example:
            "Explain why event 3 in trace 768451 takes so long compared to the prediction."

        Verbalization of outcome:
            "analyze_event_shap returns the event’s prediction, interval bounds, and SHAP feature attributions, and optionally saves visualisations that explain the contributing features."
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n🔍 Analyzing Event: TraceID={trace_id}, Position={event_position}")
        print("=" * 70)
        
        # Get the specific event
        event = self.get_event_by_position(trace_id, event_position)
        print(f"Event Details:")
        print(f"  - Activity: {event.get('activity', 'N/A')}")
        print(f"  - Machine: {event.get('machine_identifier', 'N/A')}")
        print(f"  - Product Type: {event.get('product_type', 'N/A')}")
        print(f"  - Actual Processing Time: {event[self.target_column]:.2f} minutes")
        
        # Prepare features for this event
        event_features = event[self.predictor_columns].to_frame().T
        event_processed = self._preprocess_features(event_features)
        
        # Get predictions
        main_prediction = self.main_model.predict(event_processed)[0]
        lower_bound = self.quantile_model_005.predict(event_processed)[0]
        upper_bound = self.quantile_model_095.predict(event_processed)[0]
        
        print(f"\nPredictions:")
        print(f"  - Point Prediction: {main_prediction:.2f} minutes")
        print(f"  - 90% Prediction Interval: [{lower_bound:.2f}, {upper_bound:.2f}] minutes")
        print(f"  - Interval Width: {upper_bound - lower_bound:.2f} minutes")
        
        # Calculate SHAP values
        print("\nCalculating SHAP explanations...")
        shap_values_main = self.main_explainer.shap_values(event_processed)
        shap_values_lower = self.quantile_explainer_005.shap_values(event_processed)
        shap_values_upper = self.quantile_explainer_095.shap_values(event_processed)
        
        # Create comprehensive plots
        self._create_shap_plots(
            event=event,
            event_processed=event_processed,
            shap_values_main=shap_values_main[0],
            shap_values_lower=shap_values_lower[0],
            shap_values_upper=shap_values_upper[0],
            main_prediction=main_prediction,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            trace_id=trace_id,
            event_position=event_position,
            show_plots=show_plots,
            save_plots=save_plots,
            output_dir=output_dir
        )
        
        # Prepare results dictionary
        results = {
            'trace_id': trace_id,
            'event_position': event_position,
            'event_details': {
                'activity': event.get('activity', 'N/A'),
                'machine': event.get('machine_identifier', 'N/A'),
                'product_type': event.get('product_type', 'N/A'),
                'actual_time': event[self.target_column]
            },
            'predictions': {
                'point_prediction': main_prediction,
                'lower_bound_90': lower_bound,
                'upper_bound_90': upper_bound,
                'interval_width': upper_bound - lower_bound
            },
            'shap_values': {
                'main_model': dict(zip(self.predictor_columns, shap_values_main)),
                'lower_quantile': dict(zip(self.predictor_columns, shap_values_lower)),
                'upper_quantile': dict(zip(self.predictor_columns, shap_values_upper))
            }
        }
        
        print("✓ SHAP analysis completed successfully!")
        return results
        
    def _create_shap_plots(self, event, event_processed, shap_values_main, shap_values_lower, 
                          shap_values_upper, main_prediction, lower_bound, upper_bound,
                          trace_id, event_position, show_plots, save_plots, output_dir):
        """Create comprehensive SHAP visualization plots."""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'SHAP Analysis: TraceID={trace_id}, Event Position={event_position}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Main Model SHAP Waterfall Plot
        ax1 = axes[0, 0]
        self._plot_shap_waterfall(shap_values_main, main_prediction, 
                                 'Point Prediction', ax1)
        
        # 2. Prediction Interval SHAP Analysis
        ax2 = axes[0, 1]
        self._plot_prediction_interval_shap_analysis(shap_values_lower, shap_values_upper,
                                                    lower_bound, upper_bound, ax2)
        
        # 3. Relative Feature Importance Comparison
        ax3 = axes[1, 0]
        self._plot_relative_feature_importance_comparison(shap_values_main, shap_values_lower, 
                                                        shap_values_upper, ax3)
        
        # 4. Prediction Interval Analysis
        ax4 = axes[1, 1]
        self._plot_prediction_interval_analysis(event, main_prediction, lower_bound, 
                                               upper_bound, ax4)
        
        plt.tight_layout()
        
        if save_plots:
            filename = f"{output_dir}/shap_analysis_trace_{trace_id}_event_{event_position}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✓ Plots saved to: {filename}")
            
        if show_plots:
            plt.show()
        else:
            plt.close()
            
    def _plot_shap_waterfall(self, shap_values, prediction, title, ax):
        """Create an enhanced SHAP waterfall plot showing contribution flow."""
        # Get expected value (base value) from the explainer
        try:
            base_value = self.main_explainer.expected_value
        except:
            # Fallback to mean prediction if expected_value not available
            base_value = prediction - np.sum(shap_values)
        
        # Get top 10 most important features by absolute SHAP value
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-10:]
        
        # Sort by SHAP value (most negative to most positive)
        sorted_indices = sorted(top_indices, key=lambda i: shap_values[i])
        
        top_shap = [shap_values[i] for i in sorted_indices]
        top_features = [self.predictor_columns[i] for i in sorted_indices]
        
        # Create waterfall data
        # Start with base value, then add each contribution
        positions = ['Base Value'] + top_features + ['Final Prediction']
        values = [base_value] + top_shap + [prediction]
        
        # Calculate cumulative positions for waterfall effect
        cumulative = [base_value]
        for val in top_shap:
            cumulative.append(cumulative[-1] + val)
        cumulative.append(prediction)
        
        # Create the waterfall plot
        y_pos = np.arange(len(positions))
        
        # Plot base value
        ax.barh(0, base_value, color='lightgray', alpha=0.8, height=0.6, 
               edgecolor='black', linewidth=1)
        ax.text(base_value/2, 0, f'{base_value:.1f}', ha='center', va='center', fontweight='bold')
        
        # Plot feature contributions
        for i, (feat, shap_val) in enumerate(zip(top_features, top_shap), 1):
            start_pos = cumulative[i-1]
            # Manufacturing context: negative = time decrease (good, green), positive = time increase (undesired, orange)
            color = 'lightgreen' if shap_val < 0 else 'lightsalmon'
            alpha = min(0.9, abs(shap_val) / max(abs_shap) * 0.4 + 0.6)  # Higher base alpha for better visibility
            
            ax.barh(i, shap_val, left=start_pos, color=color, alpha=alpha, height=0.6, 
                   edgecolor='black', linewidth=1)
            
            # Add contribution text
            text_pos = start_pos + shap_val/2
            ax.text(text_pos, i, f'{shap_val:+.2f}', ha='center', va='center', 
                   fontsize=9, fontweight='bold', color='white')
            
            # Add connector lines to show flow
            if i > 1:
                ax.plot([cumulative[i-2], start_pos], [i-0.4, i-0.4], 'k--', alpha=0.3, linewidth=1)
        
        # Plot final prediction
        final_idx = len(positions) - 1
        ax.barh(final_idx, prediction, color='darkblue', alpha=0.8, height=0.6,
               edgecolor='black', linewidth=1)
        ax.text(prediction/2, final_idx, f'{prediction:.1f}', ha='center', va='center', 
               fontweight='bold', color='white')
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels([pos if len(pos) <= 15 else pos[:12]+'...' for pos in positions], fontsize=9)
        ax.set_xlabel('Processing Time (minutes)', fontsize=11)
        ax.set_title(f'{title} - SHAP Waterfall\nFlow from Base Value to Final Prediction', 
                    fontsize=11, fontweight='bold')
        
        # Add vertical line at zero for reference
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # Add grid
        ax.grid(True, alpha=0.2, axis='x')
        
        # Set limits with padding
        all_values = [base_value, prediction] + [cumulative[i] for i in range(len(cumulative))]
        x_min, x_max = min(all_values), max(all_values)
        padding = (x_max - x_min) * 0.1
        ax.set_xlim(x_min - padding, x_max + padding)
        
        # Add legend with manufacturing-appropriate colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgray', alpha=0.8, label='Base Value'),
            Patch(facecolor='lightsalmon', alpha=0.7, label='Time Increase (+)'),
            Patch(facecolor='lightgreen', alpha=0.7, label='Time Decrease (-)'),
            Patch(facecolor='darkblue', alpha=0.8, label='Final Prediction')
        ]
        ax.legend(handles=legend_elements, loc='center right', fontsize=9)
        
    def _plot_prediction_interval_shap_analysis(self, shap_lower, shap_upper, lower_bound, upper_bound, ax):
        """Create converging SHAP waterfall plots that visually represent the prediction interval."""
        
        # Calculate base values for waterfall plots
        try:
            base_value_lower = self.quantile_explainer_005.expected_value
            base_value_upper = self.quantile_explainer_095.expected_value
        except:
            base_value_lower = lower_bound - np.sum(shap_lower)
            base_value_upper = upper_bound - np.sum(shap_upper)
        
        # Get top 6 most important features for cleaner visualization
        combined_abs = np.abs(shap_lower) + np.abs(shap_upper)
        top_indices = np.argsort(combined_abs)[-6:]
        
        # Sort by combined contribution magnitude for consistent ordering
        sorted_indices = sorted(top_indices, key=lambda i: combined_abs[i], reverse=True)
        
        top_features = [self.predictor_columns[i] for i in sorted_indices]
        lower_vals = [shap_lower[i] for i in sorted_indices]
        upper_vals = [shap_upper[i] for i in sorted_indices]
        
        # Create the converging waterfall layout
        n_features = len(top_features)
        total_height = 2 * n_features + 3  # Total height for both waterfalls
        
        ### LOWER WATERFALL - Flows upward from bottom ###
        lower_positions, lower_labels = self._plot_converging_waterfall(
            ax=ax,
            shap_values=lower_vals,
            features=top_features,
            base_value=base_value_lower,
            prediction=lower_bound,
            start_y=0,
            direction='upward',
            color_scheme='lower',
            target_boundary=upper_bound  # Where it should visually connect to
        )
        
        ### UPPER WATERFALL - Flows downward from top ###
        upper_positions, upper_labels = self._plot_converging_waterfall(
            ax=ax,
            shap_values=upper_vals,
            features=top_features,
            base_value=base_value_upper,
            prediction=upper_bound,
            start_y=total_height,
            direction='downward', 
            color_scheme='upper',
            target_boundary=lower_bound  # Where it should visually connect to
        )
        
        # Create dedicated y-axis element for prediction interval segment
        interval_width = upper_bound - lower_bound
        center_y = total_height / 2
        
        # Add the center interval position and label
        center_position = [center_y]
        center_label = ['Prediction Interval']
        
        # Combine all positions and labels for the axis
        all_positions = lower_positions + center_position + upper_positions
        all_labels = lower_labels + center_label + upper_labels
        
        # Set y-axis labeling
        ax.set_yticks(all_positions)
        ax.set_yticklabels(all_labels, fontsize=8)
        ax.set_ylim(-0.5, total_height + 0.5)
        
        # Add horizontal separator lines
        separator_top = center_y + 0.75
        separator_bottom = center_y - 0.75
        ax.axhline(y=separator_top, color='black', linestyle='-', alpha=0.4, linewidth=1)
        ax.axhline(y=separator_bottom, color='black', linestyle='-', alpha=0.4, linewidth=1)
        
        # Create the interval segment with width information
        ax.barh(center_y, interval_width, left=lower_bound, height=1.0, 
               color='lightsteelblue', alpha=0.6, edgecolor='black', linewidth=1)
        
        # Add interval width text on the interval bar itself
        interval_center = (lower_bound + upper_bound) / 2
        ax.text(interval_center, center_y, f'Interval Width: {interval_width:.1f} min', 
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        # Formatting
        ax.set_xlabel('Processing Time (minutes)', fontsize=11)
        ax.set_title('Prediction Interval SHAP Analysis\nConverging Waterfalls Showing Interval Formation', 
                    fontsize=12, fontweight='bold')
        
        # Add section labels
        ax.text(0.98, 0.8, 'Upper Boundary\n(flows downward)', transform=ax.transAxes,
               ha='right', va='center', fontsize=10, fontweight='bold', 
               color='darkgreen', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.text(0.98, 0.2, 'Lower Boundary\n(flows upward)', transform=ax.transAxes,
               ha='right', va='center', fontsize=10, fontweight='bold', 
               color='darkorange', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    def _plot_converging_waterfall(self, ax, shap_values, features, base_value, prediction, 
                                  start_y, direction, color_scheme, target_boundary):
        """Create a waterfall that flows either upward or downward to visually represent interval formation."""
        
        # Sort features by SHAP value for waterfall effect
        sorted_pairs = sorted(zip(shap_values, features), key=lambda x: x[0])
        sorted_shap = [pair[0] for pair in sorted_pairs]
        sorted_features = [pair[1] for pair in sorted_pairs]
        
        # Create positions for this waterfall
        positions = ['Base Value'] + sorted_features + ['Final']
        n_items = len(positions)
        
        # Calculate y-positions based on direction
        if direction == 'upward':
            y_positions = [start_y + i for i in range(n_items)]
        else:  # downward
            y_positions = [start_y - i for i in range(n_items)]
        
        # Calculate cumulative values for waterfall
        cumulative = [base_value]
        for val in sorted_shap:
            cumulative.append(cumulative[-1] + val)
        cumulative.append(prediction)
        
        # Plot base value
        base_color = 'lightcyan' if color_scheme == 'upper' else 'mistyrose'
        ax.barh(y_positions[0], base_value, color=base_color, alpha=0.8, height=0.6, 
               edgecolor='black', linewidth=1)
        ax.text(base_value/2, y_positions[0], f'{base_value:.1f}', 
               ha='center', va='center', fontweight='bold', fontsize=8)
        
        # Plot feature contributions
        max_abs_shap = max(abs(val) for val in sorted_shap) if sorted_shap else 1
        
        for i, (feature, shap_val) in enumerate(zip(sorted_features, sorted_shap), 1):
            start_pos = cumulative[i-1]
            # Manufacturing context: negative = time decrease (good, green), positive = time increase (undesired, orange)
            color = 'lightgreen' if shap_val < 0 else 'lightsalmon'
            alpha = min(0.9, abs(shap_val) / max_abs_shap * 0.4 + 0.6)  # Higher base alpha for better visibility
            
            ax.barh(y_positions[i], shap_val, left=start_pos, color=color, alpha=alpha, 
                   height=0.6, edgecolor='black', linewidth=1)
            
            # Add contribution text
            if abs(shap_val) > max_abs_shap * 0.05:  # Only show text for significant contributions
                text_pos = start_pos + shap_val/2
                ax.text(text_pos, y_positions[i], f'{shap_val:+.1f}', 
                       ha='center', va='center', fontsize=8, fontweight='bold', 
                       color='white' if alpha > 0.6 else 'black')
            
            # Add connector lines
            if i > 1:
                line_y = y_positions[i] + (0.35 if direction == 'upward' else -0.35)
                ax.plot([cumulative[i-2], start_pos], [line_y, line_y], 
                       'k--', alpha=0.3, linewidth=1)
        
        # Plot final prediction - this extends to show the interval connection
        final_color = 'darkgreen' if color_scheme == 'upper' else 'darkorange'
        
        # For visual interval representation, extend the final bar to connect with the other boundary
        if color_scheme == 'upper':
            # Upper boundary: extend down to meet lower boundary
            bar_width = prediction - target_boundary
            bar_left = target_boundary
        else:
            # Lower boundary: extend up to meet upper boundary  
            bar_width = target_boundary - prediction
            bar_left = prediction
            
        ax.barh(y_positions[-1], bar_width, left=bar_left, color=final_color, alpha=0.6, height=0.6,
               edgecolor='black', linewidth=1)
        
        # Add the actual prediction point
        ax.barh(y_positions[-1], 20, left=prediction-10, color=final_color, alpha=1.0, height=0.6,
               edgecolor='black', linewidth=1)
        ax.text(prediction, y_positions[-1], f'{prediction:.1f}', 
               ha='center', va='center', fontweight='bold', color='white', fontsize=8)
        
        # Prepare labels for return
        truncated_labels = [pos if len(pos) <= 12 else pos[:9]+'...' for pos in positions]
        
        # Add reference line
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax.grid(True, alpha=0.2, axis='x')
        
        # Return positions and labels for the calling method to manage
        return y_positions, truncated_labels
        
    def _plot_relative_feature_importance_comparison(self, shap_main, shap_lower, shap_upper, ax):
        """Plot relative feature importance comparison across all models."""
        # Get absolute SHAP values for importance
        abs_main = np.abs(shap_main)
        abs_lower = np.abs(shap_lower)
        abs_upper = np.abs(shap_upper)
        
        # Calculate relative importance (percentage of total absolute SHAP)
        total_main = np.sum(abs_main)
        total_lower = np.sum(abs_lower)
        total_upper = np.sum(abs_upper)
        
        rel_main = (abs_main / total_main * 100) if total_main > 0 else abs_main * 0
        rel_lower = (abs_lower / total_lower * 100) if total_lower > 0 else abs_lower * 0
        rel_upper = (abs_upper / total_upper * 100) if total_upper > 0 else abs_upper * 0
        
        # Get top 10 features by combined relative importance
        combined_rel_importance = rel_main + rel_lower + rel_upper
        top_indices = np.argsort(combined_rel_importance)[-10:]
        
        # Sort by average relative importance for better visualization
        sorted_indices = sorted(top_indices, key=lambda i: (rel_main[i] + rel_lower[i] + rel_upper[i]) / 3, reverse=True)
        
        top_features = [self.predictor_columns[i] for i in sorted_indices]
        main_rel = [rel_main[i] for i in sorted_indices]
        lower_rel = [rel_lower[i] for i in sorted_indices]
        upper_rel = [rel_upper[i] for i in sorted_indices]
        
        y_pos = np.arange(len(top_features))
        width = 0.25
        
        # Create grouped bar chart
        bars1 = ax.barh(y_pos - width, main_rel, width, label='Point Prediction Model', 
                       color='#2E86C1', alpha=0.8)
        bars2 = ax.barh(y_pos, lower_rel, width, label='Lower Quantile (5%)', 
                       color='#E67E22', alpha=0.8)
        bars3 = ax.barh(y_pos + width, upper_rel, width, label='Upper Quantile (95%)', 
                       color='#28B463', alpha=0.8)
        
        # Add value labels on bars for clarity
        for i, (bar, val) in enumerate(zip(bars1, main_rel)):
            if val > 1:  # Only show labels for bars with >1% importance
                ax.text(val + 0.2, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
                       va='center', fontsize=8, fontweight='bold')
        
        for i, (bar, val) in enumerate(zip(bars2, lower_rel)):
            if val > 1:
                ax.text(val + 0.2, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
                       va='center', fontsize=8, fontweight='bold')
        
        for i, (bar, val) in enumerate(zip(bars3, upper_rel)):
            if val > 1:
                ax.text(val + 0.2, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
                       va='center', fontsize=8, fontweight='bold')
        
        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f[:18] + '...' if len(f) > 18 else f for f in top_features], fontsize=9)
        ax.set_xlabel('Relative Feature Importance (%)', fontsize=11)
        ax.set_title('Relative SHAP Feature Importance Across Models\n(% of Total Absolute SHAP per Model)', 
                    fontsize=11, fontweight='bold')
        
        # Enhance legend
        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add a reference line at 10% importance
        ax.axvline(x=10, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(10.5, len(top_features)-1, '10%', rotation=90, va='bottom', 
               color='red', fontsize=8, alpha=0.7)
        
        # Set reasonable x-axis limits
        max_val = max(max(main_rel), max(lower_rel), max(upper_rel))
        ax.set_xlim(0, max_val * 1.15)
        
    def _plot_prediction_interval_analysis(self, event, main_pred, lower_bound, upper_bound, ax):
        """Analyze the prediction interval for this event."""
        actual_time = event[self.target_column]
        interval_width = upper_bound - lower_bound
        
        # Create visualization
        ax.barh([0], [interval_width], height=0.3, left=[lower_bound], 
               color='lightblue', alpha=0.7, label='90% Prediction Interval')
        ax.plot([main_pred], [0], 'bo', markersize=10, label='Point Prediction')
        ax.plot([actual_time], [0], 'ro', markersize=10, label='Actual Time')
        
        # Add text annotations
        ax.text(main_pred, 0.2, f'Pred: {main_pred:.1f}', ha='center', fontsize=10)
        ax.text(actual_time, -0.2, f'Actual: {actual_time:.1f}', ha='center', fontsize=10)
        
        # Check if actual is within interval
        within_interval = lower_bound <= actual_time <= upper_bound
        coverage_text = "✓ Within Interval" if within_interval else "✗ Outside Interval"
        color = 'green' if within_interval else 'red'
        
        ax.text(0.02, 0.95, coverage_text, transform=ax.transAxes, fontsize=12,
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Processing Time (minutes)', fontsize=12)
        ax.set_title(f'Prediction Interval Analysis\nWidth: {interval_width:.1f} min', 
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yticks([])
        
    def list_available_traces(self, limit: int = 20) -> pd.DataFrame:
        """Summarise traces present in the held-out test dataset.

        Args:
            limit (int): Maximum number of trace rows to include in the output.

        Returns:
            pandas.DataFrame: Aggregated statistics per trace (event count,
            processing time metrics, uniqueness of activities/machines).

        Example:
            >>> explainer = ManufacturingModelExplainer()
            >>> summary = explainer.list_available_traces(limit=5)
            >>> summary.columns
            Index(['TraceID', 'Event_Count', ...], dtype='object')

        User request prompt example:
            "Give me an overview of the traces available for analysis."

        Verbalization of outcome:
            "list_available_traces returns a compact table describing each trace’s event count and timing statistics so the analyst can choose which traces to explore further."
        """
        trace_info = self.test_data.groupby('TraceID').agg({
            'trace_position': 'count',
            self.target_column: ['mean', 'std', 'min', 'max'],
            'activity': lambda x: x.nunique(),
            'machine_identifier': lambda x: x.nunique()
        }).round(2)
        
        trace_info.columns = ['Event_Count', 'Avg_Time', 'Std_Time', 'Min_Time', 
                             'Max_Time', 'Unique_Activities', 'Unique_Machines']
        
        return trace_info.head(limit).reset_index()

    def _create_prediction_datasets(self, output_dir: str = "bde_analysis_output/prediction_datasets",
                                  save_format: str = 'csv') -> Dict[str, pd.DataFrame]:
        """Internal helper that materialises enhanced datasets with cached predictions."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Creating enhanced datasets with pre-computed predictions...")
        print("=" * 70)
        
        # Load the original datasets
        print("Loading original datasets...")
        train_file = os.path.join(self.data_dir, "bde_train_dataset.csv")
        val_file = os.path.join(self.data_dir, "bde_validation_dataset.csv")
        test_file = os.path.join(self.data_dir, "bde_test_dataset.csv")
        
        datasets = {
            'train': pd.read_csv(train_file, low_memory=False),
            'validation': pd.read_csv(val_file, low_memory=False),
            'test': pd.read_csv(test_file, low_memory=False)
        }
        
        enhanced_datasets = {}
        
        for dataset_name, dataset in datasets.items():
            print(f"\nProcessing {dataset_name} dataset ({len(dataset):,} events)...")
            
            # Create a copy of the dataset
            enhanced_df = dataset.copy()
            
            # Prepare features for prediction
            print(f"  - Preparing features...")
            event_features = dataset[self.predictor_columns].copy()
            event_processed = self._preprocess_features(event_features)
            
            # Make predictions in batches to manage memory
            batch_size = 10000
            n_batches = (len(event_processed) + batch_size - 1) // batch_size
            
            print(f"  - Making predictions in {n_batches} batches...")
            
            point_predictions = []
            lower_bounds = []
            upper_bounds = []
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(event_processed))
                batch_data = event_processed[start_idx:end_idx]
                
                # Get predictions for this batch
                batch_point = self.main_model.predict(batch_data)
                batch_lower = self.quantile_model_005.predict(batch_data)
                batch_upper = self.quantile_model_095.predict(batch_data)
                
                point_predictions.extend(batch_point)
                lower_bounds.extend(batch_lower)
                upper_bounds.extend(batch_upper)
                
                if (i + 1) % 5 == 0 or i == n_batches - 1:
                    print(f"    Batch {i+1}/{n_batches} completed")
            
            # Add prediction columns
            print(f"  - Adding prediction columns...")
            enhanced_df['predicted_time'] = point_predictions
            enhanced_df['prediction_lower_90'] = lower_bounds
            enhanced_df['prediction_upper_90'] = upper_bounds
            enhanced_df['prediction_interval_width'] = np.array(upper_bounds) - np.array(lower_bounds)
            enhanced_df['prediction_interval_center'] = (np.array(upper_bounds) + np.array(lower_bounds)) / 2
            
            # Calculate prediction metrics
            actual_values = enhanced_df[self.target_column].values
            enhanced_df['prediction_error'] = actual_values - enhanced_df['predicted_time']
            enhanced_df['prediction_abs_error'] = np.abs(enhanced_df['prediction_error'])
            enhanced_df['prediction_squared_error'] = enhanced_df['prediction_error'] ** 2
            enhanced_df['prediction_percentage_error'] = (enhanced_df['prediction_error'] / actual_values) * 100
            enhanced_df['prediction_abs_percentage_error'] = np.abs(enhanced_df['prediction_percentage_error'])
            
            # Coverage analysis
            enhanced_df['within_prediction_interval'] = (
                (actual_values >= enhanced_df['prediction_lower_90']) & 
                (actual_values <= enhanced_df['prediction_upper_90'])
            )
            
            # Prediction quality categories
            enhanced_df['prediction_quality'] = pd.cut(
                enhanced_df['prediction_abs_percentage_error'],
                bins=[0, 10, 25, 50, 100, float('inf')],
                labels=['Excellent (<10%)', 'Good (10-25%)', 'Fair (25-50%)', 'Poor (50-100%)', 'Very Poor (>100%)']
            )
            
            # Interval quality categories  
            enhanced_df['interval_quality'] = pd.cut(
                enhanced_df['prediction_interval_width'],
                bins=[0, 30, 60, 120, 300, float('inf')],
                labels=['Very Narrow (<30min)', 'Narrow (30-60min)', 'Medium (60-120min)', 'Wide (120-300min)', 'Very Wide (>300min)']
            )
            
            # Add percentile ranks for predictions
            enhanced_df['predicted_time_percentile'] = enhanced_df['predicted_time'].rank(pct=True) * 100
            enhanced_df['actual_time_percentile'] = enhanced_df[self.target_column].rank(pct=True) * 100
            enhanced_df['error_percentile'] = enhanced_df['prediction_abs_error'].rank(pct=True) * 100
            
            enhanced_datasets[dataset_name] = enhanced_df
            
            # Save the enhanced dataset
            if save_format.lower() == 'csv':
                output_file = os.path.join(output_dir, f"enhanced_{dataset_name}_dataset.csv")
                enhanced_df.to_csv(output_file, index=False)
            elif save_format.lower() == 'parquet':
                output_file = os.path.join(output_dir, f"enhanced_{dataset_name}_dataset.parquet")
                enhanced_df.to_parquet(output_file, index=False)
            
            print(f"  ✓ Enhanced {dataset_name} dataset saved: {output_file}")
            
            # Print summary statistics
            coverage = enhanced_df['within_prediction_interval'].mean()
            mae = enhanced_df['prediction_abs_error'].mean()
            mape = enhanced_df['prediction_abs_percentage_error'].mean()
            mean_width = enhanced_df['prediction_interval_width'].mean()
            
            print(f"  📊 Summary Statistics:")
            print(f"     Coverage: {coverage:.1%}")
            print(f"     MAE: {mae:.2f} minutes")
            print(f"     MAPE: {mape:.1f}%")
            print(f"     Mean Interval Width: {mean_width:.1f} minutes")
        
        # Create summary report
        self._create_prediction_summary_report(enhanced_datasets, output_dir)
        
        print(f"\n✅ Enhanced datasets created successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📈 Use these datasets for faster analysis - predictions are pre-computed!")
        
        return enhanced_datasets
    
    def _create_prediction_summary_report(self, enhanced_datasets: Dict[str, pd.DataFrame], output_dir: str):
        """Create a comprehensive summary report of prediction performance across all datasets."""
        
        print(f"\nCreating prediction summary report...")
        
        summary_data = []
        
        for dataset_name, df in enhanced_datasets.items():
            summary = {
                'Dataset': dataset_name.title(),
                'N_Events': len(df),
                'Coverage_90': df['within_prediction_interval'].mean(),
                'MAE': df['prediction_abs_error'].mean(),
                'RMSE': np.sqrt(df['prediction_squared_error'].mean()),
                'MAPE': df['prediction_abs_percentage_error'].mean(),
                'Median_APE': df['prediction_abs_percentage_error'].median(),
                'Mean_Interval_Width': df['prediction_interval_width'].mean(),
                'Median_Interval_Width': df['prediction_interval_width'].median(),
                'Q1_Interval_Width': df['prediction_interval_width'].quantile(0.25),
                'Q3_Interval_Width': df['prediction_interval_width'].quantile(0.75),
                'Excellent_Predictions_Pct': (df['prediction_quality'] == 'Excellent (<10%)').mean() * 100,
                'Good_Plus_Predictions_Pct': (df['prediction_quality'].isin(['Excellent (<10%)', 'Good (10-25%)'])).mean() * 100,
                'Narrow_Intervals_Pct': (df['interval_quality'].isin(['Very Narrow (<30min)', 'Narrow (30-60min)'])).mean() * 100,
                'Wide_Intervals_Pct': (df['interval_quality'].isin(['Wide (120-300min)', 'Very Wide (>300min)'])).mean() * 100
            }
            summary_data.append(summary)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_file = os.path.join(output_dir, "prediction_performance_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        
        # Create and save detailed text report
        report_file = os.path.join(output_dir, "prediction_analysis_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("ENHANCED DATASETS - PREDICTION ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATASET OVERVIEW:\n")
            f.write("-" * 20 + "\n")
            for _, row in summary_df.iterrows():
                f.write(f"{row['Dataset']} Dataset: {row['N_Events']:,} events\n")
            
            f.write("\nPREDICTION PERFORMANCE SUMMARY:\n")
            f.write("-" * 35 + "\n")
            f.write(f"{'Metric':<25} {'Train':<12} {'Validation':<12} {'Test':<12}\n")
            f.write("-" * 61 + "\n")
            
            metrics = ['Coverage_90', 'MAE', 'RMSE', 'MAPE', 'Mean_Interval_Width']
            metric_names = ['Coverage (90%)', 'MAE (minutes)', 'RMSE (minutes)', 'MAPE (%)', 'Mean Width (min)']
            
            for metric, name in zip(metrics, metric_names):
                values = [summary_df[summary_df['Dataset'] == ds][metric].iloc[0] 
                         for ds in ['Train', 'Validation', 'Test']]
                
                if metric == 'Coverage_90':
                    f.write(f"{name:<25} {values[0]:.1%}       {values[1]:.1%}       {values[2]:.1%}\n")
                elif metric in ['MAE', 'RMSE', 'Mean_Interval_Width']:
                    f.write(f"{name:<25} {values[0]:.1f}        {values[1]:.1f}        {values[2]:.1f}\n")
                else:  # MAPE
                    f.write(f"{name:<25} {values[0]:.1f}%        {values[1]:.1f}%        {values[2]:.1f}%\n")
            
            f.write("\nPREDICTION QUALITY DISTRIBUTION:\n")
            f.write("-" * 35 + "\n")
            for _, row in summary_df.iterrows():
                f.write(f"\n{row['Dataset']} Dataset:\n")
                f.write(f"  Excellent predictions (<10% error): {row['Excellent_Predictions_Pct']:.1f}%\n")
                f.write(f"  Good+ predictions (<25% error): {row['Good_Plus_Predictions_Pct']:.1f}%\n")
                f.write(f"  Narrow intervals (<60min): {row['Narrow_Intervals_Pct']:.1f}%\n")
                f.write(f"  Wide intervals (>120min): {row['Wide_Intervals_Pct']:.1f}%\n")
            
            f.write("\nKEY INSIGHTS:\n")
            f.write("-" * 15 + "\n")
            best_coverage = summary_df.loc[summary_df['Coverage_90'].idxmax(), 'Dataset']
            best_mae = summary_df.loc[summary_df['MAE'].idxmin(), 'Dataset']
            narrowest_intervals = summary_df.loc[summary_df['Mean_Interval_Width'].idxmin(), 'Dataset']
            
            f.write(f"• Best coverage: {best_coverage} dataset\n")
            f.write(f"• Lowest MAE: {best_mae} dataset\n")
            f.write(f"• Narrowest intervals: {narrowest_intervals} dataset\n")
            
            # Check for potential overfitting
            train_mae = summary_df[summary_df['Dataset'] == 'Train']['MAE'].iloc[0]
            test_mae = summary_df[summary_df['Dataset'] == 'Test']['MAE'].iloc[0]
            mae_ratio = test_mae / train_mae
            
            if mae_ratio > 1.2:
                f.write(f"⚠ Potential overfitting detected (Test MAE {mae_ratio:.1f}x Train MAE)\n")
            else:
                f.write(f"✓ Good generalization (Test MAE {mae_ratio:.1f}x Train MAE)\n")
        
        print(f"✓ Summary report saved: {report_file}")
        print(f"✓ Performance summary saved: {summary_file}")


def quick_shap_analysis(trace_id: str, event_position: int, 
                       show_plots: bool = True, save_plots: bool = True):
    """Convenience wrapper around :meth:`analyze_event_shap` for quick usage.

    Args:
        trace_id (str): Identifier of the trace containing the event of interest.
        event_position (int): Zero-based index of the event.
        show_plots (bool): Display plots inline when ``True``.
        save_plots (bool): Write plots to disk when ``True``.

    Returns:
        Dict[str, Any]: Same structure as :meth:`ManufacturingModelExplainer.analyze_event_shap`.

    Example:
        >>> quick_shap_analysis("768451", 0, show_plots=False)
        {...}

    User request prompt example:
        "Quickly explain event 0 of trace 768451 without plotting anything."

    Verbalization of outcome:
        "quick_shap_analysis instantiates the explainer, runs `analyze_event_shap`, and returns the structured SHAP analysis bundle for the requested event."
    """
    explainer = ManufacturingModelExplainer()
    return explainer.analyze_event_shap(trace_id, event_position, show_plots, save_plots)


if __name__ == "__main__":
    # Example usage
    print("Manufacturing Model Explainer - SHAP Analysis")
    print("=" * 50)
    
    # Initialize explainer
    explainer = ManufacturingModelExplainer()
    
    # Show available traces
    print("\nAvailable Traces (sample):")
    traces = explainer.list_available_traces(10)
    print(traces)
    
    # Example analysis (use first available trace)
    if not traces.empty:
        example_trace = traces.iloc[0]['TraceID']
        print(f"\nRunning example analysis on TraceID: {example_trace}")
        results = explainer.analyze_event_shap(example_trace, 0, show_plots=False)
        print("Example completed successfully!") 