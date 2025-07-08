"""
Cross-platform colored output utility using colorama
"""

import os
import sys
import colorama
from colorama import Fore, Back, Style


# Initialize colorama for cross-platform support
colorama.init(autoreset=True)
USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None


class Colors:
    """Color definitions for different output types"""

    # Main categories
    DEBUG = Fore.GREEN + Style.BRIGHT
    FLIGHT_CHECK = Fore.CYAN + Style.BRIGHT
    CHAT = Fore.WHITE + Style.NORMAL
    SYSTEM = Fore.YELLOW + Style.BRIGHT
    ERROR = Fore.RED + Style.BRIGHT
    SUCCESS = Fore.GREEN + Style.NORMAL
    WARNING = Fore.YELLOW + Style.NORMAL

    # Flight check specific
    TEST_PASS = Fore.GREEN + Style.NORMAL
    TEST_FAIL = Fore.RED + Style.NORMAL
    TEST_SKIP = Fore.YELLOW + Style.NORMAL
    TEST_TIMEOUT = Fore.MAGENTA + Style.NORMAL

    # DSPy optimization
    DSPY_INFO = Fore.BLUE + Style.BRIGHT
    OPTIMIZATION = Fore.MAGENTA + Style.BRIGHT

    # Tool operations
    TOOL_CALL = Fore.CYAN + Style.NORMAL
    TOOL_RESPONSE = Fore.BLUE + Style.NORMAL

    # Special formatting
    HEADER = Fore.WHITE + Back.BLUE + Style.BRIGHT
    SEPARATOR = Fore.BLUE + Style.DIM

    # Reset
    RESET = Style.RESET_ALL


def colored_print(text: str, color: str = Colors.CHAT, end: str = "\n"):
    """Print text with specified color"""
    if USE_COLOR:
        print(f"{color}{text}{Colors.RESET}", end=end)
    else:
        print(text, end=end)


def debug_print(text: str):
    """Print debug messages in green"""
    colored_print(f"[DEBUG] {text}", Colors.DEBUG)


def flight_check_print(text: str):
    """Print flight check messages in cyan"""
    colored_print(text, Colors.FLIGHT_CHECK)


def system_print(text: str):
    """Print system messages in yellow"""
    colored_print(f"[SYSTEM] {text}", Colors.SYSTEM)


def error_print(text: str):
    """Print error messages in red"""
    colored_print(f"[ERROR] {text}", Colors.ERROR)


def success_print(text: str):
    """Print success messages in green"""
    colored_print(f"[SUCCESS] {text}", Colors.SUCCESS)


def warning_print(text: str):
    """Print warning messages in yellow"""
    colored_print(f"[WARNING] {text}", Colors.WARNING)


def dspy_print(text: str):
    """Print DSPy-related messages in blue"""
    colored_print(f"[DSPy] {text}", Colors.DSPY_INFO)


def optimization_print(text: str):
    """Print optimization messages in magenta"""
    colored_print(f"[OPTIMIZE] {text}", Colors.OPTIMIZATION)


def tool_call_print(text: str):
    """Print tool call messages in cyan"""
    colored_print(f"[TOOL] {text}", Colors.TOOL_CALL)


def tool_response_print(text: str):
    """Print tool response messages in blue"""
    colored_print(f"[RESPONSE] {text}", Colors.TOOL_RESPONSE)


def header_print(text: str):
    """Print header with background"""
    colored_print(f" {text} ", Colors.HEADER)


def separator_print(char: str = "=", length: int = 50):
    """Print colored separator"""
    colored_print(char * length, Colors.SEPARATOR)


def test_result_print(test_name: str, result: str, time_taken: float):
    """Print test result with appropriate color"""
    color_map = {
        "PASS": Colors.TEST_PASS,
        "FAIL": Colors.TEST_FAIL,
        "SKIP": Colors.TEST_SKIP,
        "TIMEOUT": Colors.TEST_TIMEOUT,
    }

    color = color_map.get(result, Colors.CHAT)
    colored_print(f"  {test_name}... {result} ({time_taken:.2f}s)", color)


def chat_input_print(prompt: str = "Query: "):
    """Print chat input prompt"""
    colored_print(prompt, Colors.CHAT, end="")


def test_colors():
    """Test function to see all colors"""
    print("\nColor Test:")
    debug_print("This is a debug message")
    flight_check_print("This is a flight check message")
    system_print("This is a system message")
    error_print("This is an error message")
    success_print("This is a success message")
    warning_print("This is a warning message")
    dspy_print("This is a DSPy message")
    optimization_print("This is an optimization message")
    tool_call_print("This is a tool call message")
    tool_response_print("This is a tool response message")
    header_print("THIS IS A HEADER")
    separator_print()
    test_result_print("test_example", "PASS", 1.23)
    test_result_print("test_example", "FAIL", 2.45)
    test_result_print("test_example", "SKIP", 0.00)
    test_result_print("test_example", "TIMEOUT", 30.0)
    print()


if __name__ == "__main__":
    test_colors()
