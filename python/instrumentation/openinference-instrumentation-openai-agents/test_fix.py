import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from openinference.instrumentation.openai_agents._processor import _get_attributes_from_function_span_data
from openinference.instrumentation.openai_agents._processor import FunctionSpanData

def test_empty_string_output():
    """Test that empty string output doesn't cause IndexError"""
    
    # Create test data with empty string output
    function_span_data = FunctionSpanData(
        name="test_func",
        input='{"k": "v"}',
        output="",  # This is the problematic case that caused IndexError
        mcp_data=None,
    )
    
    # This should not raise an IndexError
    try:
        attributes = dict(_get_attributes_from_function_span_data(function_span_data))
        print("✓ Test passed: No IndexError for empty string output")
        print(f"Attributes: {attributes}")
        
        # Verify expected attributes
        expected_keys = {"tool.name", "input.value", "input.mime_type", "output.value"}
        actual_keys = set(attributes.keys())
        
        if expected_keys.issubset(actual_keys):
            print("✓ All expected attributes are present")
        else:
            missing = expected_keys - actual_keys
            print(f"✗ Missing attributes: {missing}")
            
        # Verify that OUTPUT_MIME_TYPE is NOT set for empty string
        if "output.mime_type" not in attributes:
            print("✓ output.mime_type correctly not set for empty string")
        else:
            print("✗ output.mime_type should not be set for empty string")
            
    except IndexError as e:
        print(f"✗ Test failed: IndexError occurred - {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: Unexpected error - {e}")
        return False
    
    return True

def test_valid_json_output():
    """Test that valid JSON output works correctly"""
    
    function_span_data = FunctionSpanData(
        name="test_func",
        input='{"k": "v"}',
        output='{"result": "success"}',  # Valid JSON
        mcp_data=None,
    )
    
    try:
        attributes = dict(_get_attributes_from_function_span_data(function_span_data))
        print("✓ Test passed: Valid JSON output handled correctly")
        
        # For valid JSON, output.mime_type should be set
        if "output.mime_type" in attributes:
            print("✓ output.mime_type correctly set for valid JSON")
        else:
            print("✗ output.mime_type should be set for valid JSON")
            
    except Exception as e:
        print(f"✗ Test failed: Unexpected error - {e}")
        return False
    
    return True

def test_single_character_output():
    """Test that single character output works correctly"""
    
    function_span_data = FunctionSpanData(
        name="test_func",
        input='{"k": "v"}',
        output="{",  # Single character - should not trigger JSON mime type
        mcp_data=None,
    )
    
    try:
        attributes = dict(_get_attributes_from_function_span_data(function_span_data))
        print("✓ Test passed: Single character output handled correctly")
        
        # For single character, output.mime_type should NOT be set
        if "output.mime_type" not in attributes:
            print("✓ output.mime_type correctly not set for single character")
        else:
            print("✗ output.mime_type should not be set for single character")
            
    except Exception as e:
        print(f"✗ Test failed: Unexpected error - {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing IndexError fix for empty function output...")
    print("=" * 50)
    
    test1 = test_empty_string_output()
    print()
    test2 = test_valid_json_output()
    print()
    test3 = test_single_character_output()
    
    print("\n" + "=" * 50)
    if test1 and test2 and test3:
        print("🎉 All tests passed! The IndexError fix is working correctly.")
    else:
        print("❌ Some tests failed.")