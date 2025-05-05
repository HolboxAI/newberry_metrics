# Newberry Metrics

A Python package for tracking and analyzing AWS Bedrock API usage metrics, including costs and latency.

## Latest Version: 0.1.4

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

### 1. Initialize TokenEstimator

Initialize the tracker with a specific Bedrock model ID. You can also optionally provide cost and latency thresholds for automatic alerting.

```python
from newberry_metrics import TokenEstimator
import json # For printing examples

# Initialize with your model ID
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

# Optional: Define alert thresholds
cost_alert_threshold = 0.05  # Alert if total session cost exceeds $0.05
latency_alert_threshold_ms = 2000 # Alert if any single call takes > 2000ms

estimator = TokenEstimator(
    model_id=model_id,
    cost_threshold=cost_alert_threshold,      # Optional
    latency_threshold_ms=latency_alert_threshold_ms # Optional
)
```

### 2. Get Model Pricing

Retrieve the cost per million tokens for the initialized model.

```python
costs = estimator.get_model_cost_per_million()
print(f"Input cost per million: ${costs['input']}")
print(f"Output cost per million: ${costs['output']}")
```

### 3. Making API Calls & Tracking Metrics

The `_invoke_bedrock` method (though marked private) handles calling the Bedrock model. Crucially, it **automatically updates and saves the session metrics** (cost, latency, token counts) after each call.

If the thresholds set during initialization are exceeded (total cost or current call latency), alerts will be printed to the console during this step.

```python
# Make an API call - session metrics are automatically tracked & alerts checked
prompt = "Explain the concept of Large Language Models."
# Note: Using the private _invoke_bedrock method here as shown in main.py example
# A public wrapper might be added in future versions.
response = estimator._invoke_bedrock(prompt, max_tokens=200)

current_call_metrics = response['SessionMetrics']['current_call']

answer = response['SessionMetrics'].get('answer', '(Answer not found)') # Answer is also included
print(f"\n--- Single Call Results ---")
print(f"Answer: {answer[:100]}...") # Print truncated answer
print(f"Cost (This Call): ${current_call_metrics['cost']:.6f}")
print(f"Latency (This Call): {current_call_metrics['latency']:.3f}s")
print(f"Tokens (In/Out): {current_call_metrics['input_tokens']}/{current_call_metrics['output_tokens']}")


# Get session metrics from the response
session_metrics = response['SessionMetrics']
print(f"Total session cost: ${session_metrics['total_cost']}")
print(f"Average cost: ${session_metrics['average_cost']}")
print(f"Total latency: {session_metrics['total_latency']} seconds")
print(f"Average latency: {session_metrics['average_latency']} seconds")
print(f"Total calls: {session_metrics['total_calls']}")

# Access the session metrics updated by this call (contained within the response)
session_metrics_after_call = response['SessionMetrics']
print(f"Latest total session cost: ${session_metrics_after_call['total_cost']:.6f}")
print(f"Cost of current call: ${session_metrics_after_call['current_call']['cost']:.6f}")
```

### 4. Retrieve Current Session Metrics

You can get the complete metrics object for the current session (reflecting all calls made so far) at any time. This reads the latest state managed by the estimator.

```python
from dataclasses import asdict # For printing example

current_metrics = estimator.get_session_metrics()
print(f"Total calls so far: {current_metrics.total_calls}")
print(f"Average latency: {current_metrics.average_latency:.3f}s")

```

### 5. Reset Session Metrics

Reset the tracked metrics for the current session (identified by AWS credentials) back to zero in the persistent store (DynamoDB).

```python
estimator.reset_session_metrics()
print("Session metrics have been reset.")
```

### 6. Visualize Session Metrics (Optional)

Generate bar charts summarizing the cost and average latency of the session's API calls, grouped either hourly or daily. Requires `matplotlib` and `pandas` to be installed.

```python
# Ensure you have data first by making some calls
# response = estimator._invoke_bedrock("Another prompt...")

try:
    # Show plots grouped by hour (default)
    print("Displaying hourly metrics plot...")
    estimator.visualize_metrics(time_interval='hourly')

    # Or, show plots grouped by day
    print("Displaying daily metrics plot...")
    estimator.visualize_metrics(time_interval='daily')

    # Or, save the plots to files instead of showing them
    # Creates 'session_plots_hourly.png' and 'session_plots_daily.png'
    # print("Saving metrics plots...")
    # estimator.visualize_metrics(time_interval='hourly', save_path='session_plots')
    # estimator.visualize_metrics(time_interval='daily', save_path='session_plots')

except ImportError:
    print("Please install matplotlib and pandas to use visualization: pip install matplotlib pandas")
except Exception as e:
    print(f"An error occurred during visualization: {e}")
```
```

2.  **Update the "Requirements" section to:**

```markdown
## Requirements
- Python >= 3.10
- `boto3` for AWS Bedrock integration
- `matplotlib` for visualization
- `pandas` for visualization
```

These changes add the documentation for the `visualize_metrics` method and ensure the necessary dependencies are listed. Remember to update the version in `setup.py` if you release this.

### Optional: Analyzing a Specific Response (Advanced)

If you have a raw response object from `_invoke_bedrock` (or elsewhere), you can calculate its specific cost/latency independently using these helper methods. Note that this calculation is already performed internally by `_invoke_bedrock` during the tracking process.

## Supported Models

The package includes pricing information for the following Bedrock models (primarily in `us-east-1`). Ensure the model ID you use matches one of these or add its pricing to `get_model_cost_per_million` if needed.

- amazon.nova-pro-v1:0 ($0.003/$0.012 per 1K tokens)
- amazon.nova-micro-v1:0 ($0.000035/$0.00014 per 1K tokens)
- anthropic.claude-3-sonnet-20240229-v1:0 ($0.003/$0.015 per 1K tokens)
- anthropic.claude-3-haiku-20240307-v1:0 ($0.00025/$0.00125 per 1K tokens)
- anthropic.claude-3-opus-20240229-v1:0 ($0.015/$0.075 per 1K tokens)
- meta.llama2-13b-chat-v1 ($0.00075/$0.001 per 1K tokens)
- meta.llama2-70b-chat-v1 ($0.00195/$0.00256 per 1K tokens)
- ai21.jamba-1-5-large-v1:0 ($0.0125/$0.0125 per 1K tokens)
- cohere.command-r-v1:0 ($0.0005/$0.0015 per 1K tokens)
- cohere.command-r-plus-v1:0 ($0.003/$0.015 per 1K tokens)
- mistral.mistral-7b-instruct-v0:2 ($0.0002/$0.0006 per 1K tokens)
- mistral.mixtral-8x7b-instruct-v0:1 ($0.0007/$0.0021 per 1K tokens)
*(Pricing based on us-east-1, may vary in other regions)*

## Session Metrics & Alerting

The package automatically tracks and persists session metrics using **Amazon DynamoDB**. A dedicated table (default name: `BedrockSessionMetrics`) is required in your AWS account in the specified region. Each session's data is stored as an item keyed by a unique hash derived from the AWS credentials and region used.

Metrics stored include:
- `total_cost`: Cumulative cost for the session.
- `average_cost`: Average cost per call in the session.
- `total_latency`: Cumulative latency (in seconds) for the session.
- `average_latency`: Average latency per call in the session.
- `total_calls`: Total number of API calls made in the session.
- `api_calls`: A detailed list (`List[APICallMetrics]`) of each individual API call containing its timestamp, cost, latency, token counts, and call number within the session.

**Alerting:**
If `cost_threshold` (float, e.g., `0.10` for $0.10) or `latency_threshold_ms` (float, e.g., `1500.0` for 1500ms) are provided during `TokenEstimator` initialization, the package will automatically print warning messages to the console if:
- The **total cost** for the current session exceeds the `cost_threshold` after an API call.
- The **latency** of any individual API call exceeds the `latency_threshold_ms`.

## Recent Updates


## Requirements
- Python >= 3.10
- `boto3` for AWS Bedrock integration
- `matplotlib` for visualization
- `pandas` for visualization

## Contact & Support
- **Developer**: Satya-Holbox, Harshika-Holbox
- **Email**: satyanarayan@holbox.ai
- **GitHub**: [SatyaTheG](https://github.com/SatyaTheG)

## License
This project is licensed under the MIT License.

---

**Note**: This package is actively maintained and regularly updated with new features and model support.
