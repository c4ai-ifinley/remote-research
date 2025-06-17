#!/usr/bin/env python3
"""
Standalone test script for DSPy configuration
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_dspy_setup():
    """Test DSPy setup independently"""
    print("=" * 50)
    print("DSPy Standalone Test")
    print("=" * 50)

    # Check environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("BASE_URL")

    print(f"OPENAI_API_KEY: {'✓ Found' if openai_key else '✗ Missing'}")
    print(f"BASE_URL: {base_url if base_url else '✗ Missing'}")

    if not openai_key:
        print("\nError: OPENAI_API_KEY not found in environment")
        print("Please add it to your .env file")
        return False

    # Test DSPy import
    try:
        import dspy

        print("✓ DSPy import successful")
    except ImportError as e:
        print(f"✗ DSPy import failed: {e}")
        print("Try: uv add dspy-ai")
        return False

    # Test DSPy configuration
    try:
        print("\nTesting DSPy configuration...")

        # For custom OpenAI-compatible endpoints, use "openai/" prefix
        # o3-mini requires specific parameters
        config = {
            "model": "openai/o3-mini-birthright",  # Add openai/ prefix
            "api_key": openai_key,
            "max_tokens": 20000,  # Required minimum for reasoning models
            "temperature": 1.0,  # Required for reasoning models
        }

        if base_url:
            config["base_url"] = base_url

        print(f"Config: {config}")

        lm = dspy.LM(**config)
        dspy.configure(lm=lm)

        print("✓ DSPy LM configured successfully")

        # Test a simple call
        class SimpleSignature(dspy.Signature):
            question = dspy.InputField()
            answer = dspy.OutputField()

        simple_qa = dspy.ChainOfThought(SimpleSignature)

        print("\nTesting simple DSPy call...")
        result = simple_qa(question="What is 2+2?")
        print(f"✓ DSPy call successful! Answer: {result.answer}")

        return True

    except Exception as e:
        print(f"✗ DSPy configuration failed: {e}")
        import traceback

        print(f"Full error:\n{traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = test_dspy_setup()
    if success:
        print("\n🎉 DSPy is working correctly!")
    else:
        print("\n❌ DSPy setup failed - check the errors above")
