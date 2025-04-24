# Newberry Metrics

A Python package for tracking and estimating AI model token costs and usage metrics.

## Latest Version: 0.0.10

## Features

### Cost Tracking and Estimation
- Model cost calculation per million tokens
- Prompt cost estimation
- Session-based cost tracking
- Support for multiple AI models:
  - Claude 3.7 Sonnet
  - Nova Micro

## Installation

```bash
pip install newberry_metrics
```

## Usage Examples

### Calculate Model Cost
```python
from newberry_metrics import model_cost

# Get cost per million tokens for Nova Micro
nova_cost = model_cost("nova micro")
print(f"Nova Micro cost per million tokens: ${nova_cost}")

# Get cost for Claude 3.7 Sonnet
claude_cost = model_cost("claude 3.7 sonnet")
print(f"Claude 3.7 Sonnet cost per million tokens: ${claude_cost}")
```

### Estimate Prompt Cost
```python
from newberry_metrics import prompt_cost

prompt = "What is the weather in San Francisco?"
cost = prompt_cost(prompt, model="nova micro")
print(f"Estimated prompt cost: ${cost}")
```

### Track Session Costs
```python
from newberry_metrics import session_cost

# Track costs across multiple prompts in a session
session_id = "session_1"
cost1 = session_cost(session_id, "First prompt", model="nova micro")
cost2 = session_cost(session_id, "Second prompt", model="nova micro")
print(f"Total session cost: ${cost2}")  # Shows cumulative cost
```

## Technical Details

### Token Estimation
- Uses a simple 4 characters per token estimation rule
- Provides conservative estimates for cost calculation
- Supports both input and output token cost calculation

## Recent Updates (v0.0.10)

### New Features
- Implemented basic token cost estimation
- Added support for Claude 3.7 Sonnet and Nova Micro models
- Introduced session-based cost tracking
- Added utility functions for cost estimation

### Technical Improvements
- Simple token estimation algorithm
- Session cost persistence during runtime
- Model pricing configuration system

## Requirements
- Python >= 3.10

## Contact & Support
- **Developer**: Satya-Holbox, Harshika-Holbox
- **Email**: satyanarayan@holbox.ai
- **GitHub**: [SatyaTheG](https://github.com/SatyaTheG)

## License
This project is licensed under the MIT License.

---

**Note**: This package is actively maintained and regularly updated with new features and model support.
