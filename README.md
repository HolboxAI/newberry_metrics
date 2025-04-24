# Newberry Metrics

A Python package for tracking and estimating AI model token costs and usage metrics.

## Latest Version: 0.1.0

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

- **Initialize TokenEstimator**
```python
from newberry_metrics import TokenEstimator

estimator = TokenEstimator(
    model_id="amazon.nova-pro-v1:0",
    aws_access_key_id="YOUR_AWS_ACCESS_KEY_ID",
    aws_secret_access_key="YOUR_AWS_SECRET_ACCESS_KEY",
    region="us-east-1"
)
```

- **Get Model Cost per Million Tokens**
```python
costs = estimator.get_model_cost_per_million()
print(f"Input cost per million: ${costs['input']}")
print(f"Output cost per million: ${costs['output']}")
```

- **Calculate Prompt Cost**
```python
# Invoke the model to get a response object
response = estimator._invoke_bedrock("What is the weather in San Francisco?")

# Calculate cost based on the response object obtained by invoking the bedrock model
result = estimator.calculate_prompt_cost(response)
print(f"Total cost: ${result['cost']}")
print(f"Input tokens: {result['input_tokens']} (cost: ${result['input_cost']})")
print(f"Output tokens: {result['output_tokens']} (cost: ${result['output_cost']})")
print(f"Answer: {result['answer']}")
```

- **Track Session Costs**
```python
session_id = "session_1"
session_info = estimator.track_session_cost(session_id, response)
print(f"Cumulative session cost: ${session_info['session_cost']}")
```

## Technical Details


## Recent Updates (v0.1.0)

### New Features
- Introduced class-based `TokenEstimator` API replacing standalone functions.
- Amazon Bedrock integration for real-time answers and token usage.
- Separate input/output token cost calculation with configurable model pricing.
- AWS credentials and region now provided at initialization.

### Technical Improvements
- Improved code organization and maintainability with data classes and type hints.
- Detailed response parsing and error handling for unconventional formats.
- Session-based cost accumulation supports multi-prompt workflows.

## Requirements
- Python >= 3.10
- `boto3` for AWS Bedrock integration

## Contact & Support
- **Developer**: Satya-Holbox, Harshika-Holbox
- **Email**: satyanarayan@holbox.ai
- **GitHub**: [SatyaTheG](https://github.com/SatyaTheG)

## License
This project is licensed under the MIT License.

---

**Note**: This package is actively maintained and regularly updated with new features and model support.
