# Newberry Metrics

A Python package for tracking and analyzing AWS Bedrock API usage metrics, including costs and latency.

## Latest Version: 0.1.0

## Features

- Track API call costs and latency
- Monitor token usage (input and output)
- Maintain session-based metrics
- Support for multiple Bedrock models
- Automatic AWS credential handling
- Detailed latency tracking and analysis

## Installation

```bash
pip install newberry_metrics
```

## AWS Credential Setup

The package uses the AWS credential chain to authenticate with AWS services. You can set up credentials in one of the following ways:

### 1. Using IAM Role (Recommended for EC2)
- Attach an IAM role to your EC2 instance with Bedrock permissions
- No additional configuration needed
- The code will automatically use the instance's IAM role credentials

### 2. Using AWS CLI
```bash
aws configure
```
This will create a credentials file at `~/.aws/credentials` with your access key and secret key.

### 3. Using Environment Variables
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=your_region
```

## Usage Examples

### Initialize TokenEstimator
```python
from newberry_metrics import TokenEstimator

# Initialize with your model ID
estimator = TokenEstimator(model_id="anthropic.claude-3-sonnet-20240229-v1:0")
```

### Get Model Cost per Million Tokens
```python
costs = estimator.get_model_cost_per_million()
print(f"Input cost per million: ${costs['input']}")
print(f"Output cost per million: ${costs['output']}")
```

### Calculate Prompt Cost and Latency
```python
# Make an API call - session metrics are automatically tracked
response = estimator._invoke_bedrock("What is the weather in San Francisco?")

# Get session metrics from the response
session_metrics = response['SessionMetrics']
print(f"Total session cost: ${session_metrics['total_cost']}")
print(f"Average cost: ${session_metrics['average_cost']}")
print(f"Total latency: {session_metrics['total_latency']} seconds")
print(f"Average latency: {session_metrics['average_latency']} seconds")
print(f"Total calls: {session_metrics['total_calls']}")

# Calculate cost for this specific call
cost_info = estimator.calculate_prompt_cost(response)
print(f"Total cost: ${cost_info['cost']}")
print(f"Input tokens: {cost_info['input_tokens']} (cost: ${cost_info['input_cost']})")
print(f"Output tokens: {cost_info['output_tokens']} (cost: ${cost_info['output_cost']})")
print(f"Answer: {cost_info['answer']}")

# Calculate latency for this specific call
latency_info = estimator.calculate_prompt_latency(response)
print(f"Latency: {latency_info['latency_seconds']} seconds")
print(f"Latency: {latency_info['latency_milliseconds']} milliseconds")
```

### Track Session Metrics
```python
# Session metrics are automatically tracked with each API call
# You can still access the current session metrics at any time
metrics = estimator.get_session_metrics()
print(f"Current session metrics: {metrics}")

# Reset session metrics if needed
estimator.reset_session_metrics()
```

### Session Management
```python
# Get all session metrics
metrics = estimator.get_session_metrics()

# Reset session metrics
estimator.reset_session_metrics()
```

## Supported Models

The package supports the following Bedrock models with their respective pricing:

- amazon.nova-pro-v1:0 ($0.003/$0.012 per 1K tokens)
- amazon.nova-micro-v1:0 ($0.000035/$0.00014 per 1K tokens)
- anthropic.claude-3-sonnet-20240229-v1:0 ($0.003/$0.015 per 1K tokens)
- anthropic.claude-3-haiku-20240307-v1:0 ($0.00025/$0.00125 per 1K tokens)
- meta.llama2-13b-chat-v1 ($0.00075/$0.001 per 1K tokens)
- meta.llama2-70b-chat-v1 ($0.00195/$0.00256 per 1K tokens)

## Session Metrics

The package automatically tracks and stores session metrics in JSON files. Each session is identified by a hash of the AWS credentials used. Metrics include:

- Total cost
- Average cost
- Total latency
- Average latency
- Total API calls
- Detailed API call history with timestamps

## Recent Updates (v0.1.0)

### New Features
- Added latency tracking and analysis
- Automatic AWS credential handling
- Improved session metrics with latency tracking
- Enhanced cost tracking with separate input/output calculations

### Technical Improvements
- Better code organization with separate response processing
- Improved error handling for API responses
- More efficient session cost tracking
- Better separation of concerns in the codebase

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
