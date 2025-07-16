#!/usr/bin/env python3
"""
Configuration setup script for Enhanced Generic Flight Checker
Run this to create all necessary configuration files
"""

import json
import os
from pathlib import Path


def create_enhanced_flight_config():
    """Create enhanced flight check configuration"""
    config = {
        "metadata": {
            "version": "2.0.0-enhanced",
            "created": "2025-01-08",
            "description": "Enhanced Generic MCP Flight Checker with Dynamic Dependency Analysis",
        },
        "flight_checker": {
            "enabled": True,
            "run_on_startup": True,
            "default_verbosity": "normal",
            "timeout_multiplier": 1.0,
            "max_optimization_attempts": 3,
            "auto_export_reports": True,
            "parallel_execution": False,
        },
        "dspy_config": {
            "optimization_enabled": True,
            "model": "openai/o3-mini-birthright",
            "max_tokens": 20000,
            "temperature": 0.8,
            "fallback_to_rules": True,
            "dependency_analysis_enabled": True,
            "test_generation_enabled": True,
        },
        "dependency_analysis": {
            "enabled": True,
            "use_dspy": True,
            "confidence_threshold": 0.5,
            "max_dependency_depth": 3,
            "auto_discovery": True,
            "fallback_patterns": {
                "search_provides_data": ["search", "extract", "get", "info"],
                "list_enables_read": ["list", "directory", "read", "file"],
                "fetch_provides_content": ["fetch", "extract", "process"],
            },
        },
        "test_generation": {
            "auto_generate": True,
            "use_dspy": True,
            "pattern_matching_fallback": True,
            "min_test_coverage": 0.8,
            "generate_negative_tests": False,
            "adapt_to_failures": True,
        },
        "test_criteria": {
            "default_timeout": 30.0,
            "min_response_length": 5,
            "acceptable_error_patterns": [
                "no saved information",
                "not found",
                "file not found",
                "no information available",
            ],
            "required_success_patterns": {
                "search_papers": ["paper", "arxiv"],
                "extract_info": ["title", "author"],
                "read_file": ["content", "json"],
                "list_directory": ["file", "dir"],
                "fetch": ["content", "response"],
            },
        },
        "learning": {
            "save_successful_tests": True,
            "learned_tests_file": "learned_flight_tests.json",
            "max_learned_tests_per_tool": 10,
            "adaptive_criteria": True,
            "success_threshold": 0.8,
            "learning_rate": 0.1,
        },
        "reporting": {
            "auto_export": True,
            "export_format": "json",
            "include_dependency_graph": True,
            "include_optimization_history": True,
            "compress_large_responses": True,
            "max_response_preview_length": 200,
        },
    }

    config_file = "enhanced_flight_config.json"
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Created enhanced flight check configuration: {config_file}")
    else:
        print(f"Configuration file already exists: {config_file}")

    return config_file


def create_test_cases_template():
    """Create enhanced test cases template"""
    test_cases = {
        "metadata": {
            "version": "2.0.0",
            "description": "Enhanced test cases with dynamic generation capabilities",
            "last_updated": "2025-01-08",
        },
        "test_cases": {
            "search_papers": [
                {
                    "test_name": "basic_search_test",
                    "description": "Test basic paper search functionality",
                    "prompt": "Search for academic papers about machine learning. Return exactly 2 arXiv paper IDs.",
                    "expected_indicators": ["paper", "arxiv", "id"],
                    "expected_format": "list_of_ids",
                    "timeout_seconds": 30.0,
                    "critical": True,
                    "success_criteria": {
                        "min_response_length": 15,
                        "required_patterns": ["paper"],
                        "acceptable_not_found": False,
                    },
                    "optimization_history": [],
                }
            ],
            "extract_info": [
                {
                    "test_name": "paper_extraction_test",
                    "description": "Test paper information extraction",
                    "prompt": "Extract detailed information about paper with ID 'test_paper_123'. Show title, authors, summary, and publication details. If no paper is found, clearly state that no information is available.",
                    "expected_indicators": ["title", "author", "summary"],
                    "expected_format": "structured_text",
                    "timeout_seconds": 20.0,
                    "critical": True,
                    "success_criteria": {
                        "min_response_length": 20,
                        "acceptable_not_found": True,
                        "required_patterns": [],
                    },
                    "optimization_history": [],
                }
            ],
            "read_file": [
                {
                    "test_name": "config_file_read_test",
                    "description": "Test reading configuration files",
                    "prompt": "Read the server_config.json file and display its complete contents. Show the JSON structure clearly.",
                    "expected_indicators": ["json", "config", "content"],
                    "expected_format": "file_content",
                    "timeout_seconds": 15.0,
                    "critical": True,
                    "success_criteria": {
                        "min_response_length": 30,
                        "required_patterns": ["json", "config"],
                        "expect_json": False,
                    },
                    "optimization_history": [],
                }
            ],
            "list_directory": [
                {
                    "test_name": "current_directory_listing",
                    "description": "Test listing current directory contents",
                    "prompt": "List all files and directories in the current folder. Show both files and directories with clear indicators.",
                    "expected_indicators": ["file", "directory", "dir"],
                    "expected_format": "directory_listing",
                    "timeout_seconds": 15.0,
                    "critical": True,
                    "success_criteria": {
                        "min_response_length": 20,
                        "required_patterns": ["file"],
                        "acceptable_not_found": False,
                    },
                    "optimization_history": [],
                }
            ],
            "fetch": [
                {
                    "test_name": "web_content_fetch_test",
                    "description": "Test fetching web content",
                    "prompt": "Fetch content from https://example.com and display the returned content with proper formatting.",
                    "expected_indicators": ["content", "response", "http"],
                    "expected_format": "web_content",
                    "timeout_seconds": 30.0,
                    "critical": False,
                    "success_criteria": {
                        "min_response_length": 25,
                        "required_patterns": ["content"],
                        "acceptable_not_found": False,
                    },
                    "optimization_history": [],
                }
            ],
        },
        "prompt_templates": {
            "search_template": "Search for {topic} papers. Return {max_results} arXiv paper IDs.",
            "extract_template": "Extract information about paper {paper_id}. Show title, authors, and summary.",
            "read_template": "Read file {filepath} and display its contents clearly.",
            "list_template": "List all items in {path} with type indicators.",
            "fetch_template": "Fetch content from {url} and display the response.",
        },
        "dspy_config": {
            "optimization_enabled": True,
            "dependency_analysis_enabled": True,
            "test_generation_enabled": True,
            "adaptive_learning": True,
        },
    }

    config_file = "enhanced_test_cases.json"
    if not os.path.exists(config_file):
        with open(config_file, "w") as f:
            json.dump(test_cases, f, indent=2)
        print(f"Created enhanced test cases template: {config_file}")
    else:
        print(f"Test cases file already exists: {config_file}")

    return config_file


def create_environment_template():
    """Create environment variables template"""
    env_template = """# Enhanced Generic MCP Flight Checker Environment Configuration
# Copy this to .env and fill in your values

# Required: OpenAI API key for o3-mini model
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Custom base URL for API endpoints
BASE_URL=https://ai-incubator-api.pnnl.gov

# Required: Anthropic API key for the main chatbot
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Flight Checker Configuration
FLIGHT_CHECK_ENABLED=true
FLIGHT_CHECK_ON_STARTUP=true

# DSPy Configuration
DSPY_OPTIMIZATION_ENABLED=true
DSPY_DEPENDENCY_ANALYSIS=true
DSPY_TEST_GENERATION=true

# Debugging
DEBUG_MODE=false
VERBOSE_LOGGING=false

# Performance Tuning
MAX_CONCURRENT_TESTS=3
DEFAULT_TIMEOUT=30
OPTIMIZATION_ATTEMPTS=3

# Learning System
ENABLE_LEARNING=true
ADAPTIVE_CRITERIA=true
"""

    env_file = ".env.template"
    if not os.path.exists(env_file):
        with open(env_file, "w") as f:
            f.write(env_template)
        print(f"Created environment template: {env_file}")
        print("   Copy this to .env and update with your API keys")
    else:
        print(f"Environment template already exists: {env_file}")

    return env_file


def create_learned_tests_template():
    """Create learned tests template"""
    learned_tests = {
        "metadata": {
            "version": "2.0.0",
            "description": "Learned test cases from successful executions",
            "last_updated": "2025-01-08",
            "total_successes": 0,
        },
        "list_directory": [
            {
                "test_name": "current_directory_listing",
                "description": "Test listing current directory contents",
                "prompt": "List all files and directories in the current folder. Show me what's available.",
                "success_count": 2,
                "last_success": "2025-01-08T12:00:00.000000",
                "success_criteria": {
                    "min_response_length": 20,
                    "required_patterns": ["file"],
                },
                "optimization_history": [],
                "generation_strategy": "learned",
            }
        ],
    }

    learned_file = "learned_flight_tests.json"
    if not os.path.exists(learned_file):
        with open(learned_file, "w") as f:
            json.dump(learned_tests, f, indent=2)
        print(f"Created learned tests template: {learned_file}")
    else:
        print(f"Learned tests file already exists: {learned_file}")

    return learned_file


def create_readme():
    """Create comprehensive README for the enhanced system"""
    readme_content = """# Enhanced Generic MCP Flight Checker

## Overview

The Enhanced Generic MCP Flight Checker is an intelligent testing system that automatically discovers MCP tools, analyzes their dependencies using o3-mini, and generates comprehensive test cases with dynamic optimization.

## Key Features

-  **Auto-Discovery**: Automatically detects all available MCP tools
-  **DSPy Optimization**: Uses o3-mini for intelligent prompt optimization
-  **Dynamic Dependency Analysis**: o3-mini analyzes tool relationships automatically
-  **Function Call Verification**: Actually calls MCP tools to prevent hallucinations
-  **Learning System**: Saves successful test cases for future reuse
-  **Cross-Platform**: Works on Windows, Linux, and macOS with colored output

## Quick Start

### 1. Setup Environment

```bash
# Copy environment template
cp .env.template .env

# Edit .env with your API keys
nano .env
```

### 2. Install Dependencies

```bash
pip install dspy-ai colorama python-dotenv anthropic
```

### 3. Run Flight Check

```bash
# Basic usage
python mcp_chatbot.py

# Standalone flight check
python mcp_chatbot.py flight-check

# Health monitoring
python mcp_chatbot.py health-monitor
```

## Configuration Files

- `enhanced_flight_config.json` - Main configuration
- `enhanced_test_cases.json` - Test case templates  
- `learned_flight_tests.json` - Learned successful tests
- `.env` - Environment variables and API keys

## Commands

### In Chatbot
- `/flight-check [verbosity]` - Run comprehensive flight check
- `/flight-status` - Quick system status
- `/flight-deps` - Show dependency graph
- `/flight-tools` - Show discovered tools
- `/flight-export` - Export latest report

### Standalone Scripts
- `python mcp_chatbot.py flight-check` - Standalone flight check
- `python mcp_chatbot.py health-monitor` - Continuous monitoring
- `python mcp_chatbot.py create-config` - Create configuration
- `python mcp_chatbot.py test-dspy` - Test DSPy connection

## Verbosity Levels

- `quiet` - No output except critical errors
- `minimal` - Basic pass/fail status
- `normal` - Standard output with summaries
- `verbose` - Detailed test information
- `debug` - Full debug information

## Dependency Analysis

The system uses o3-mini to automatically discover tool relationships:

```
search_papers → extract_info (data dependency)
list_directory → read_file (prerequisite)
fetch → extract_info (optional)
```

## Test Generation

Tests are generated using:
1. **DSPy Intelligence**: o3-mini analyzes tools and creates smart tests
2. **Pattern Matching**: Fallback using name-based patterns
3. **Learning System**: Reuses previously successful tests

## Optimization

Failed tests are automatically optimized using:
- o3-mini analysis of failure reasons
- Dependency context awareness
- Previous attempt avoidance
- Tool-specific improvements

## Reports

Detailed JSON reports include:
- Test execution results
- Dependency graph analysis
- Optimization history
- Performance metrics
- System readiness status

## Troubleshooting

### DSPy Not Available
- Install: `pip install dspy-ai`
- Set `OPENAI_API_KEY` in `.env`
- System falls back to rule-based optimization

### No Tools Discovered
- Check MCP server connections
- Verify `server_config.json`
- Ensure tools are properly registered

### Tests Failing
- System automatically optimizes failed prompts
- Check dependency requirements
- Review tool-specific configurations

## Integration

The system is designed to integrate with existing MCP chatbots:

```python
from enhanced_flight_check import EnhancedFlightChecker

# Create checker
checker = EnhancedFlightChecker(chatbot)

# Run comprehensive check
report = await checker.run_flight_check()

# Get insights
insights = checker.get_dependency_insights()
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Enhanced Flight Checker                  │
├─────────────────────────────────────────────────────────┤
│  Auto-Discovery → Dependency Analysis → Test Generation │
│       ↓                    ↓                    ↓       │
│  Tool Detection    →   o3-mini Analysis  →   DSPy Tests │
│       ↓                    ↓                    ↓       │
│  Execute Tests     →   Validate Results  →   Optimize   │
│       ↓                    ↓                    ↓       │
│  Learn & Store     →   Generate Report   →   Export     │
└─────────────────────────────────────────────────────────┘
```

## File Structure

```
project/
├── enhanced_flight_check.py      # Main flight checker
├── mcp_chatbot.py                # Enhanced chatbot integration
├── dspy_optimizer.py             # Enhanced DSPy optimization
├── color_utils.py                # Cross-platform colors
├── utils.py                      # Utility functions
├── enhanced_flight_config.json   # Configuration
├── enhanced_test_cases.json      # Test templates
├── learned_flight_tests.json     # Learned tests
├── .env                          # Environment variables
└── README.md                     # This file
```

## Advanced Usage

### Custom Test Cases

```python
# Add custom test case
custom_test = DynamicTestCase(
    tool_name="search_papers",
    test_name="custom_ml_search",
    description="Search for ML papers",
    prompt="Search for recent neural network papers",
    expected_indicators=["neural", "network"],
    success_criteria={"min_response_length": 20}
)

checker.test_cases["search_papers"].append(custom_test)
```

### Dependency Analysis

```python
# Get dependency insights
insights = checker.get_dependency_insights()
print(f"Total dependencies: {insights['total_dependencies']}")
print(f"Execution order: {insights['execution_chains']}")
```

### Custom Optimization

```python
# Custom optimization context
context = OptimizationContext(
    tool_name="search_papers",
    original_prompt="Find papers",
    failure_reason="Too vague",
    expected_output_format="list",
    success_criteria={"min_length": 10},
    previous_attempts=[],
    dependency_context="Provides data for extract_info"
)

optimized = optimizer.optimize_prompt(context)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all flight checks pass
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review configuration files
3. Run diagnostic commands
4. Submit an issue with logs

---

*Enhanced Generic MCP Flight Checker v2.0.0*
"""

    readme_file = "README_ENHANCED.md"
    if not os.path.exists(readme_file):
        with open(readme_file, "w") as f:
            f.write(readme_content)
        print(f"Created comprehensive README: {readme_file}")
    else:
        print(f"README already exists: {readme_file}")

    return readme_file


def create_integration_script():
    """Create integration helper script"""
    integration_script = """#!/usr/bin/env python3
\"\"\"
Integration script for Enhanced Generic MCP Flight Checker
Use this to integrate with your existing MCP chatbot
\"\"\"

import asyncio
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from enhanced_flight_check import EnhancedFlightChecker, FlightCheckIntegration, VerbosityLevel
    from mcp_chatbot import MCP_ChatBot
    from color_utils import success_print, error_print, system_print
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required files are in the same directory")
    sys.exit(1)


async def quick_integration_test():
    \"\"\"Quick test to verify integration works\"\"\"
    system_print("Running quick integration test...")
    
    try:
        # Initialize chatbot
        chatbot = MCP_ChatBot()
        await chatbot.connect_to_servers()
        
        if not chatbot.available_tools:
            error_print("No tools discovered - check your MCP server configuration")
            return False
        
        success_print(f"Discovered {len(chatbot.available_tools)} tools")
        
        # Test flight checker creation
        checker = EnhancedFlightChecker(chatbot)
        success_print("Enhanced flight checker created successfully")
        
        # Test dependency analysis
        insights = checker.get_dependency_insights()
        success_print(f"Dependency analysis complete: {insights['total_dependencies']} relationships")
        
        # Quick flight check
        ready = await FlightCheckIntegration.run_quick_check(chatbot)
        
        if ready:
            success_print("  Integration test passed - system ready!")
        else:
            error_print("   Some issues detected - run full flight check for details")
        
        await chatbot.cleanup()
        return ready
        
    except Exception as e:
        error_print(f"Integration test failed: {e}")
        return False


async def full_integration_test():
    \"\"\"Full integration test with comprehensive flight check\"\"\"
    system_print("Running full integration test...")
    
    try:
        # Initialize chatbot
        chatbot = MCP_ChatBot()
        await chatbot.connect_to_servers()
        
        # Run comprehensive flight check
        report = await FlightCheckIntegration.run_comprehensive_check(
            chatbot, export_report=True
        )
        
        # Print results
        if report.system_ready:
            success_print("   Full integration test passed!")
            success_print(f"  Tests: {report.passed}/{report.total_tests} passed")
            success_print(f"  Dependencies: {sum(len(deps) for deps in report.dependency_graph.values())} discovered")
            success_print(f"  Optimization: {'Applied' if report.optimization_applied else 'Not needed'}")
        else:
            error_print(" Integration test found issues:")
            error_print(f"  Critical failures: {report.critical_failures}")
            error_print(f"  Dependency failures: {report.dependency_failures}")
        
        await chatbot.cleanup()
        return report.system_ready
        
    except Exception as e:
        error_print(f"Full integration test failed: {e}")
        return False


def show_integration_options():
    \"\"\"Show integration options\"\"\"
    print("Enhanced Generic MCP Flight Checker - Integration Helper")
    print("=" * 60)
    print()
    print("Available commands:")
    print("  quick-test     - Quick integration verification")
    print("  full-test      - Comprehensive integration test")
    print("  setup-config   - Create configuration files")
    print("  check-deps     - Check dependencies")
    print("  show-tools     - Show discovered tools")
    print()
    print("Usage: python integration.py <command>")


async def setup_configuration():
    \"\"\"Setup configuration files\"\"\"
    system_print("Setting up configuration files...")
    
    from config_setup import (
        create_enhanced_flight_config,
        create_test_cases_template,
        create_environment_template,
        create_learned_tests_template
    )
    
    create_enhanced_flight_config()
    create_test_cases_template()
    create_environment_template()
    create_learned_tests_template()
    
    success_print("Configuration setup complete!")
    print()
    print("Next steps:")
    print("1. Copy .env.template to .env")
    print("2. Add your API keys to .env")
    print("3. Run: python integration.py quick-test")


async def check_dependencies():
    \"\"\"Check if all dependencies are available\"\"\"
    system_print("Checking dependencies...")
    
    missing = []
    
    try:
        import dspy
        success_print("DSPy available")
    except ImportError:
        missing.append("dspy-ai")
        error_print("DSPy not available")
    
    try:
        import colorama
        success_print("Colorama available")
    except ImportError:
        missing.append("colorama")
        error_print("Colorama not available")
    
    try:
        from dotenv import load_dotenv
        success_print("Python-dotenv available")
    except ImportError:
        missing.append("python-dotenv")
        error_print("Python-dotenv not available")
    
    try:
        import anthropic
        success_print("Anthropic available")
    except ImportError:
        missing.append("anthropic")
        error_print("Anthropic not available")
    
    if missing:
        error_print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    else:
        success_print("All dependencies available!")
        return True


async def show_discovered_tools():
    \"\"\"Show discovered MCP tools\"\"\"
    system_print("Discovering MCP tools...")
    
    try:
        chatbot = MCP_ChatBot()
        await chatbot.connect_to_servers()
        
        if chatbot.available_tools:
            success_print(f"Discovered {len(chatbot.available_tools)} tools:")
            for tool in chatbot.available_tools:
                print(f"  - {tool['name']}: {tool.get('description', 'No description')}")
        else:
            error_print("No tools discovered")
            print("Check your server_config.json and MCP server connections")
        
        await chatbot.cleanup()
        
    except Exception as e:
        error_print(f"Failed to discover tools: {e}")


async def main():
    \"\"\"Main integration helper\"\"\"
    if len(sys.argv) != 2:
        show_integration_options()
        return
    
    command = sys.argv[1].lower()
    
    if command == "quick-test":
        success = await quick_integration_test()
        sys.exit(0 if success else 1)
    elif command == "full-test":
        success = await full_integration_test()
        sys.exit(0 if success else 1)
    elif command == "setup-config":
        await setup_configuration()
    elif command == "check-deps":
        success = await check_dependencies()
        sys.exit(0 if success else 1)
    elif command == "show-tools":
        await show_discovered_tools()
    else:
        show_integration_options()
        error_print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
"""

    script_file = "integration.py"
    if not os.path.exists(script_file):
        with open(script_file, "w") as f:
            f.write(integration_script)

        # Make executable on Unix systems
        try:
            os.chmod(script_file, 0o755)
        except:
            pass  # Windows doesn't need this

        print(f"Created integration helper script: {script_file}")
        print("   Usage: python integration.py <command>")
    else:
        print(f"Integration script already exists: {script_file}")

    return script_file


def main():
    """Main configuration setup function"""
    print("Enhanced Generic MCP Flight Checker - Configuration Setup")
    print("=" * 60)
    print()

    # Create all configuration files
    create_enhanced_flight_config()
    create_test_cases_template()
    create_environment_template()
    create_learned_tests_template()
    create_readme()
    create_integration_script()

    print()
    print("Configuration setup complete!")
    print()
    print("Next steps:")
    print("1. Copy .env.template to .env and add your API keys")
    print("2. Run: python integration.py check-deps")
    print("3. Run: python integration.py quick-test")
    print("4. Run: python mcp_chatbot.py")
    print()
    print("For help: python integration.py")


if __name__ == "__main__":
    main()
