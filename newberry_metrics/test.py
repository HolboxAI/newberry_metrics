# test.py

from newberry_metrics.main import APICallMetrics, SessionMetrics, visualize_metrics
from datetime import datetime, timedelta
import random

# Generate sample API call data for 2 days, 48 calls (one per hour)
api_calls = []
base_time = datetime.now() - timedelta(days=2)
for i in range(48):
    call_time = base_time + timedelta(hours=i)
    api_calls.append(
        APICallMetrics(
            timestamp=call_time.isoformat(),
            cost=round(random.uniform(0.001, 0.01), 4),
            latency=round(random.uniform(0.5, 2.5), 3),
            call_counter=i + 1,
            input_tokens=random.randint(100, 500),
            output_tokens=random.randint(50, 300)
        )
    )

# Create a SessionMetrics object
metrics = SessionMetrics(
    total_cost=sum(call.cost for call in api_calls),
    average_cost=sum(call.cost for call in api_calls) / len(api_calls),
    total_latency=sum(call.latency for call in api_calls),
    average_latency=sum(call.latency for call in api_calls) / len(api_calls),
    total_calls=len(api_calls),
    api_calls=api_calls
)

# Test the visualize_metrics function
visualize_metrics(metrics)