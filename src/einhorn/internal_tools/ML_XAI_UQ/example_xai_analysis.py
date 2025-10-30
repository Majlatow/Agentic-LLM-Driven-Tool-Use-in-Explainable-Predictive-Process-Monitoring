"""
Example: Using ML_UQ_XAI_functions for Explainable AI Analysis

This script demonstrates how to use the SHAP-based explainable AI functionality
to analyze individual manufacturing events and understand model predictions.
"""

from ML_UQ_XAI_functions import ManufacturingModelExplainer, quick_shap_analysis
import pandas as pd
import numpy as np

def main():
    print("Manufacturing Model - Explainable AI Analysis Demo")
    print("=" * 60)
    print("Featuring Enhanced SHAP Waterfall Plots!")
    
    # Initialize the explainer
    print("Initializing explainer...")
    explainer = ManufacturingModelExplainer()
    
    # Show available traces for analysis
    print("\nAvailable Traces for Analysis:")
    traces = explainer.list_available_traces(15)
    print(traces)
    
    # Example 1: Analyze a specific event using the class method
    print("\n" + "="*60)
    print("Example 1: Detailed Event Analysis with Enhanced SHAP Waterfall")
    print("="*60)
    
    # Select first trace and analyze its first event
    if not traces.empty:
        example_trace = int(traces.iloc[0]['TraceID'])  # Convert to int to avoid .0 suffix
        print(f"Analyzing TraceID: {example_trace}, Event Position: 0")
        print("The enhanced waterfall plot will show:")
        print("  - Base value (model's expected prediction)")
        print("  - Feature contributions (positive/negative)")
        print("  - Cumulative flow to final prediction")
        print("  - Visual connection lines between contributions")
        
        results = explainer.analyze_event_shap(
            trace_id=str(example_trace), 
            event_position=0,
            show_plots=True,  # Display plots
            save_plots=True   # Save plots to disk
        )
        
        print(f"\nAnalysis Results:")
        print(f"  - Trace ID: {results['trace_id']}")
        print(f"  - Event Position: {results['event_position']}")
        print(f"  - Activity: {results['event_details']['activity']}")
        print(f"  - Machine: {results['event_details']['machine']}")
        print(f"  - Actual Time: {results['event_details']['actual_time']:.2f} min")
        print(f"  - Predicted Time: {results['predictions']['point_prediction']:.2f} min")
        print(f"  - Prediction Interval: [{results['predictions']['lower_bound_90']:.2f}, {results['predictions']['upper_bound_90']:.2f}] min")
        print(f"  - Interval Width: {results['predictions']['interval_width']:.2f} min")
        
        # Show top contributing features
        shap_main = results['shap_values']['main_model']
        sorted_features = sorted(shap_main.items(), key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\nTop 5 Most Important Features (SHAP values):")
        for i, (feature, shap_val) in enumerate(sorted_features[:5]):
            # Handle both scalar and array SHAP values
            if isinstance(shap_val, (list, np.ndarray)):
                shap_val_scalar = float(shap_val[0]) if len(shap_val) > 0 else 0.0
            else:
                shap_val_scalar = float(shap_val)
            
            direction = "increases" if shap_val_scalar > 0 else "decreases"
            print(f"  {i+1}. {feature}: {shap_val_scalar:.3f} ({direction} prediction)")
    
    # Example 2: Quick analysis using the convenience function
    print("\n" + "="*60)
    print("Example 2: Quick Analysis Function")
    print("="*60)
    
    if not traces.empty and len(traces) > 1:
        second_trace = int(traces.iloc[1]['TraceID'])  # Convert to int to avoid .0 suffix
        print(f"Quick analysis of TraceID: {second_trace}, Event Position: 0")
        
        # Using the quick_shap_analysis function
        quick_results = quick_shap_analysis(
            trace_id=str(second_trace),
            event_position=0,
            show_plots=False,  # Don't show plots for quick analysis
            save_plots=True    # But still save them
        )
        
        print(f"Quick analysis completed for {quick_results['trace_id']}")
    
    # Example 3: Examine multiple events from the same trace
    print("\n" + "="*60)
    print("Example 3: Multiple Events from Same Trace")
    print("="*60)
    
    if not traces.empty:
        # Find a trace with multiple events
        multi_event_trace = traces[traces['Event_Count'] >= 5].iloc[0] if len(traces[traces['Event_Count'] >= 5]) > 0 else traces.iloc[0]
        trace_id = str(int(multi_event_trace['TraceID']))  # Convert to int then str to avoid .0 suffix
        event_count = int(multi_event_trace['Event_Count'])
        
        print(f"Analyzing trace {trace_id} with {event_count} events")
        
        # Analyze first 3 events of this trace
        for event_pos in range(min(3, event_count)):
            print(f"\n  Event {event_pos + 1}/{event_count}:")
            
            event_results = explainer.analyze_event_shap(
                trace_id=trace_id,
                event_position=event_pos,
                show_plots=False,  # Don't show plots for batch analysis
                save_plots=True
            )
            
            print(f"    Activity: {event_results['event_details']['activity']}")
            print(f"    Predicted Time: {event_results['predictions']['point_prediction']:.2f} min")
            print(f"    Actual Time: {event_results['event_details']['actual_time']:.2f} min")
            
            # Check prediction accuracy
            pred_error = abs(event_results['predictions']['point_prediction'] - event_results['event_details']['actual_time'])
            print(f"    Prediction Error: {pred_error:.2f} min")
            
            # Check if actual is within prediction interval
            within_interval = (event_results['predictions']['lower_bound_90'] <= 
                             event_results['event_details']['actual_time'] <= 
                             event_results['predictions']['upper_bound_90'])
            coverage_status = "Within Interval" if within_interval else "Outside Interval"
            print(f"    Coverage: {coverage_status}")
    
    # Example 4: Demonstrate prediction dataset creation
    print("\n" + "="*60)
    print("Example 4: Enhanced Prediction Datasets")
    print("="*60)
    print("The explainer can also create enhanced datasets with pre-computed predictions:")
    print("  - Internal helper: explainer._create_prediction_datasets()")
    print("  - Creates train/validation/test datasets with:")
    print("    • Point predictions and intervals")
    print("    • Prediction errors and quality metrics")
    print("    • Coverage indicators")
    print("    • Quality categorizations")
    print("  - Enables faster future analysis (no re-computation needed)")
    print("  - Internal helper `_create_prediction_datasets` drives the generation")
    
    print("\n" + "="*60)
    print("Output Files:")
    print("  - SHAP analysis plots: bde_analysis_output/xai_analysis/")
    print("  - Enhanced waterfall plots show feature contribution flow")
    print("  - File format: shap_analysis_trace_{TraceID}_event_{Position}.png")
    print("  - Enhanced datasets: use the dedicated CLI script if available")
    print("\nXAI Analysis Demo Completed!")
    print("Enhanced features:")
    print("  ✓ Improved SHAP waterfall plots with base values and flow")
    print("  ✓ Enhanced visualization with contribution transparency")
    print("  ✓ Connector lines showing cumulative contributions")
    print("  ✓ Prediction dataset creation functionality")
    print("  ✓ Comprehensive performance analysis tools")


if __name__ == "__main__":
    main() 