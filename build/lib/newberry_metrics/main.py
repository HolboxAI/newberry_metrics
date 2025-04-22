# Cost per 1000 tokens for input/output
MODEL_PRICING = {
    "claude 3.7 sonnet": {"input": 0.003, "output": 0.015},
    "nova micro": {"input": 0.000035, "output": 0.00014},
}

# Session tracking
SESSION_COSTS = {}

# --- Token estimator (basic dummy approach) ---
def estimate_tokens(text):
    # Simple estimate: 1 token ≈ 4 characters (OpenAI-style rule of thumb)
    return max(1, len(text) // 4)

# --- Function 1: Get cost per million input tokens for a model ---
def model_cost(model):
    model_key = model.strip().lower()
    if model_key not in MODEL_PRICING:
        raise ValueError(f"Unknown model: {model}")
    input_price = MODEL_PRICING[model_key]["input"]
    return round(input_price * 1000, 6)  # Convert from per 1k to per 1M

# --- Function 2: Estimate prompt cost using default model ---
def prompt_cost(prompt, model="nova micro"):
    model_key = model.strip().lower()
    if model_key not in MODEL_PRICING:
        raise ValueError(f"Unknown model: {model}")

    input_tokens = estimate_tokens(prompt)
    output_tokens = estimate_tokens(prompt)  # Dummy, assume same output tokens

    input_cost = input_tokens * MODEL_PRICING[model_key]["input"]
    output_cost = output_tokens * MODEL_PRICING[model_key]["output"]
    return round(input_cost + output_cost, 6)

# --- Function 3: Session cost tracker ---
def session_cost(session_id, prompt, model="nova micro"):
    model_key = model.strip().lower()
    if model_key not in MODEL_PRICING:
        raise ValueError(f"Unknown model: {model}")

    input_tokens = estimate_tokens(prompt)
    output_tokens = estimate_tokens(prompt)

    input_cost = input_tokens * MODEL_PRICING[model_key]["input"]
    output_cost = output_tokens * MODEL_PRICING[model_key]["output"]
    total_cost = input_cost + output_cost

    if session_id not in SESSION_COSTS:
        SESSION_COSTS[session_id] = 0.0
    SESSION_COSTS[session_id] += total_cost

    return round(SESSION_COSTS[session_id], 6)

# --- test section ---
if __name__ == "__main__":
    print("Model Cost (Nova):", model_cost("Nova Micro"))
    print("Model Cost (Claude):", model_cost("Claude 3.7 Sonnet"))

    prompt = "What is the weather in San Francisco"
    print("Prompt Cost:", prompt_cost(prompt))

    print("Session 1 (Prompt 1):", session_cost("Session1", prompt))
    print("Session 1 (Prompt 2):", session_cost("Session1", "What is the weather in New York"))
