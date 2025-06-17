#!/usr/bin/env python3
"""
Standalone test script for DSPy configuration
"""

import os
from dotenv import load_dotenv

# Import color utilities
from color_utils import (
    debug_print,
    system_print,
    error_print,
    success_print,
    warning_print,
    dspy_print,
    header_print,
    separator_print,
    test_colors,
)

# Load environment variables
load_dotenv()


def test_dspy_setup():
    """Test DSPy setup independently"""
    separator_print()
    header_print("DSPy Standalone Test")
    separator_print()

    # Check environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL")

    system_print(f"OPENAI_API_KEY: {'✓ Found' if openai_key else '✗ Missing'}")
    system_print(f"BASE_URL: {base_url if base_url else '✗ Missing'}")

    if not openai_key:
        error_print("OPENAI_API_KEY not found in environment")
        warning_print("Please add it to your .env file")
        return False

    # Test DSPy import
    try:
        import dspy

        success_print("DSPy import successful")
    except ImportError as e:
        error_print(f"DSPy import failed: {e}")
        warning_print("Try: uv add dspy-ai")
        return False

    # Test DSPy configuration
    try:
        dspy_print("Testing DSPy configuration...")

        config = {
            "model": "openai/o3-mini-birthright",
            "api_key": openai_key,
            "max_tokens": 20000,
            "temperature": 1.0,
        }

        if base_url:
            config["base_url"] = base_url

        debug_print(f"Config: {config}")

        lm = dspy.LM(**config)
        dspy.configure(lm=lm)

        success_print("DSPy LM configured successfully")

        # Test a simple call
        class SimpleSignature(dspy.Signature):
            question = dspy.InputField()
            answer = dspy.OutputField()

        simple_qa = dspy.ChainOfThought(SimpleSignature)

        dspy_print("Testing simple DSPy call...")
        result = simple_qa(question="What is 2+2?")
        success_print(f"DSPy call successful! Answer: {result.answer}")

        return True

    except Exception as e:
        error_print(f"DSPy configuration failed: {e}")
        import traceback

        debug_print(f"Full error:\n{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    # First show color test
    test_colors()

    # Then test DSPy
    success = test_dspy_setup()
    if success:
        success_print("🎉 DSPy is working correctly!")
    else:
        error_print("❌ DSPy setup failed - check the errors above")
