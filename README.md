# Newberry Metrics

A Python package for tracking and analyzing AWS Bedrock API usage metrics, including costs, latency, and token usage, with an automatically launched dashboard for live visualization.

## Features

- Track API call costs, latency, and token usage (input/output).
- **Automatic Background Dashboard**: A Streamlit dashboard is launched as a background process upon `TokenEstimator` initialization, providing live visualization.
- **Dashboard Features**: Displays KPIs (total/average cost & latency), hourly/daily charts, and detailed call logs.
- **Persistent Session Storage**: Maintains session-based metrics in a JSON file located in `~/.newberry_metrics/sessions/`, uniquely identified by AWS credentials.
- Support for multiple Bedrock models.
- Automatic AWS credential handling.
- Console alerts for configurable cost and latency thresholds.
- Static method (`TokenEstimator.stop_dashboard()`) to manually stop the background dashboard process.
- **Duplicate Call Prevention**: Ensures metrics for a single Bedrock response are logged only once, even if `calculate_prompt_cost` is called multiple times with the same response object.

## Installation

```bash
pip install newberry_metrics
```
This will also install necessary dependencies like `streamlit`, `pandas`, and `plotly`.

## AWS Credential Setup

The package uses the AWS credential chain. Configure your AWS credentials via IAM roles (recommended for EC2), `aws configure`, or environment variables.

## Usage Examples

### 1. Initialize TokenEstimator & Launch Dashboard

When `TokenEstimator` is initialized, it automatically launches the Newberry Metrics dashboard in the background if it's not already running. The console will display the dashboard URL (typically `http://localhost:8501`).

```python
from newberry_metrics import TokenEstimator

model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
region = "us-east-1"
cost_alert_threshold = 0.05
latency_alert_threshold_ms = 2000

estimator = TokenEstimator(
    model_id=model_id,
    region=region,
    cost_threshold=cost_alert_threshold,      # Optional
    latency_threshold_ms=latency_alert_threshold_ms # Optional
)
```
The dashboard will continue running even if your script finishes. Open the URL in your browser.

### 2. Get Model Pricing

```python
costs = estimator.get_model_cost_per_million()
print(f"Input cost/1M tokens: ${costs['input']}, Output cost/1M tokens: ${costs['output']}")
```

### 3. Making API Calls & Tracking Metrics

First, get the raw response object from Bedrock using `get_response()`. Then, pass this object to `calculate_prompt_cost()` to process it, calculate metrics, and update the session file.

```python
prompt = "Explain Large Language Models simply."
max_tokens_to_generate = 150

# Step 1: Get the raw response object
raw_response_object = estimator.get_response(
    prompt=prompt, 
    max_tokens=max_tokens_to_generate
)

# Step 2: Calculate cost and update metrics
# This also returns detailed info about the call and session.
call_and_session_info = estimator.calculate_prompt_cost(raw_response_object)

print(f"Answer (truncated): {call_and_session_info.get('answer', 'N/A')[:100]}...")
current_call_metrics = call_and_session_info.get('current_call_metrics', {})
print(f"Cost for this call: ${current_call_metrics.get('cost', 0):.6f}")
print(f"Total session cost: ${call_and_session_info.get('total_cost_session', 0):.6f}")
```
Refresh your dashboard (using its refresh button 🔄) to see the new data. If `calculate_prompt_cost` is called again with the *same* `raw_response_object`, new metrics will *not* be logged, preventing duplicates.

### 4. Using the Dashboard

- **Automatic Launch**: Starts in the background with `TokenEstimator`. URL and PID are printed.
- **Persistence**: Runs independently of the launching script.
- **Data Source**: Reads from `~/.newberry_metrics/sessions/session_metrics_<CREDENTIAL_HASH>.json`.
- **Refresh Button**: Manually click the 🔄 button on the dashboard to load the latest metrics after new calls.
- **Shutdown**:
    - Programmatically: `TokenEstimator.stop_dashboard()`
    - Manually: Kill the process using the PID (from console or `~/.newberry_metrics/sessions/.newberry_dashboard.pid`).

```python
from newberry_metrics import TokenEstimator 
TokenEstimator.stop_dashboard()
```

### 5. Retrieve Current Session Metrics Programmatically

```python
current_session_object = estimator.get_session_metrics()
print(f"Total calls: {current_session_object.total_calls}, Total cost: ${current_session_object.total_cost:.6f}")
```

### 6. Reset Session Metrics

Resets metrics in the current session's JSON file to zero.

```python
estimator.reset_session_metrics()
```

## Supported Models

Pricing information is included for (but not limited to):
- amazon.nova-pro-v1:0
- anthropic.claude-3-haiku-20240307-v1:0
- anthropic.claude-3-sonnet-20240229-v1:0
- anthropic.claude-3-opus-20240229-v1:0
- anthropic.claude-3-5-sonnet-20240620-v1:0
- meta.llama2-13b-chat-v1
- meta.llama2-70b-chat-v1
- ai21.jamba-1-5-large-v1:0
- cohere.command-r-v1:0
- cohere.command-r-plus-v1:0
- mistral.mistral-7b-instruct-v0:2
- mistral.mixtral-8x7b-instruct-v0:1

Parsing logic for these and other models is in `bedrock_models.py`.

## Session Metrics & Alerting

- **Session File Location**: `~/.newberry_metrics/sessions/session_metrics_<CREDENTIAL_HASH>.json`. The hash is derived from AWS credentials and region.
- **PID File Location**: `~/.newberry_metrics/sessions/.newberry_dashboard.pid`.
- **Dashboard Data**: The Streamlit dashboard reads from the session JSON file.
- **Metrics Stored**: `total_cost`, `average_cost`, `total_latency`, `average_latency`, `total_calls`, and a detailed list `api_calls` (each with `timestamp`, `cost`, `latency`, `input_tokens`, `output_tokens`, `call_counter`).
- **Alerting**: Console warnings are printed if `cost_threshold` (total session cost) or `latency_threshold_ms` (individual call latency) are exceeded.

## Requirements
- Python >= 3.10
- `boto3`
- `streamlit`
- `pandas`
- `plotly`

## Contact & Support
- **Developer**: Satya-Holbox, Harshika-Holbox
- **Email**: satyanarayan@holbox.ai
- **GitHub**: [SatyaTheG](https://github.com/SatyaTheG)

## License
This project is licensed under the MIT License.

---

**Note**: This package is actively maintained. Please ensure you are using the latest version for new features and model support.
