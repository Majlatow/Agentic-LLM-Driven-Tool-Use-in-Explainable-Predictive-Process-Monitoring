"""
Test script for the XAI internal tool functionality.
This script demonstrates how to use the new XAI analysis functions.
"""

from xai_analysis import (
    get_available_traces,
    get_event_SHAP_explanation,
    get_event_processing_time_predictions,
    extract_SHAP_feature_importance,
    validate_event_data,
    quick_event_analysis,
)

def test_xai_functions():
    """Test the XAI functions to ensure they work correctly."""
    print("Testing XAI Internal Tool Functions")
    print("=" * 50)
    
    # Test 1: Get available traces
    print("\n1. Testing get_available_traces()...")
    traces_result = get_available_traces(limit=5)
    if traces_result["success"]:
        print(f"✓ Found {traces_result['total_traces']} traces")
        if traces_result['traces']:
            print(f"   First trace: {traces_result['traces'][0]}")
    else:
        print(f"✗ Error: {traces_result['error']}")
    
    # Test 2: Validate event data (if traces are available)
    if traces_result["success"] and traces_result['traces']:
        first_trace = traces_result['traces'][0]
        trace_id = str(first_trace.get('TraceID', '1'))
        
        print(f"\n2. Testing validate_event_data() for trace {trace_id}...")
        validation_result = validate_event_data(trace_id, 0)
        if validation_result["valid"]:
            print(f"✓ Validation successful")
            print(f"   Events in trace: {validation_result['trace_events_count']}")
            print(f"   Activity: {validation_result['event_activity']}")
            print(f"   Machine: {validation_result['event_machine']}")
        else:
            print(f"✗ Validation failed: {validation_result['error']}")
        
        # Test 3: Get event predictions
        print(f"\n3. Testing get_event_processing_time_predictions() for trace {trace_id}...")
        predictions_result = get_event_processing_time_predictions((trace_id, 0))
        if predictions_result["success"]:
            print("✓ Time prediction retrieval successful")
            print(f"   Events returned: {len(predictions_result['events'])}")
        else:
            print(f"✗ Prediction retrieval error: {predictions_result['error']}")

        print(f"\n5. Testing get_event_SHAP_explanation() for trace {trace_id}...")
        analysis_result = get_event_SHAP_explanation(trace_id, 0, show_plots=False, save_plots=False)
        if analysis_result["success"]:
            print("✓ analyze_events succeeded")
            print(f"   Events returned: {len(analysis_result['events'])}")
        else:
            print(f"✗ analyze_events error: {analysis_result['error']}")

        print(f"\n6. Testing quick_event_analysis() for trace {trace_id}...")
        quick_result = quick_event_analysis(trace_id, 0)
        if quick_result["success"]:
            print("✓ Quick event analysis successful")
        else:
            print(f"✗ Quick event analysis error: {quick_result['error']}")
        
        # Test 4: Analyze manufacturing event
        print(f"\n4. Testing get_event_SHAP_explanation() for trace {trace_id}...")
        analysis_result = get_event_SHAP_explanation(trace_id, 0, show_plots=False, save_plots=False)
        if analysis_result["success"]:
            print(f"✓ Analysis completed successfully")
            print(f"   Activity: {analysis_result['event_details']['activity']}")
            print(f"   Machine: {analysis_result['event_details']['machine']}")
            print(f"   Actual time: {analysis_result['event_details']['actual_time']:.2f} min")
            print(f"   Predicted time: {analysis_result['predictions']['point_prediction']:.2f} min")
            print(f"   Prediction error: {analysis_result['prediction_error']:.2f} min")
            print(f"   Within interval: {analysis_result['within_interval']}")
            
            # Show top features
            if analysis_result['top_features']:
                print(f"   Top contributing features:")
                for feature in analysis_result['top_features'][:3]:
                    print(f"     {feature['rank']}. {feature['feature']}: {feature['shap_value']:.3f} ({feature['direction']})")
        else:
            print(f"✗ Error in analysis: {analysis_result['error']}")
        
        # Test 5: Quick event analysis
        print(f"\n5. Testing quick_event_analysis() for trace {trace_id}...")
        quick_result = quick_event_analysis(trace_id, 0)
        if quick_result["success"]:
            print(f"✓ Quick analysis completed")
            print(f"   Analysis completed: {quick_result['analysis_completed']}")
            print(f"   Plots saved: {quick_result['plots_saved']}")
        else:
            print(f"✗ Error in quick analysis: {quick_result['error']}")
    
    # Test 6: Test extract_SHAP_feature_importance with dummy data
    print(f"\n6. Testing extract_SHAP_feature_importance() with dummy data...")
    dummy_shap_values = {
        "feature_a": 0.75,
        "feature_b": -0.45,
        "feature_c": 0.10,
    }
    top_features = extract_SHAP_feature_importance(dummy_shap_values, limit=3)
    print(f"✓ Extracted {len(top_features)} top features:")
    for feature in top_features:
        print(f"   {feature['rank']}. {feature['feature']}: {feature['shap_value']:.3f} ({feature['direction']})")
    
    print(f"\n" + "=" * 50)
    print("XAI Internal Tool Testing Completed!")

if __name__ == "__main__":
    test_xai_functions() 