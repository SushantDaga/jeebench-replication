"""
Configuration file for LLM evaluation system.
Replace the placeholder API keys with your actual keys.
"""
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# API keys for different providers
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY", "your-openai-api-key"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY", "your-anthropic-api-key"),
    "gemini": os.getenv("GEMINI_API_KEY", "your-gemini-api-key"),
    "openrouter": os.getenv("OPENROUTER_API_KEY", "your-openrouter-api-key")
}

# Default models for each provider
DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-3-7-sonnet-20250219",
    "gemini": "gemini-2.5-flash-preview-04-17",
    "openrouter": "qwen/qwen-2.5-72b-instruct:free"
}

# Default parameters for API calls
DEFAULT_PARAMS = {
    "openai": {
        "temperature": 0.0,
        "max_tokens": 1000,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": None
    },
    "anthropic": {
        "temperature": 0.0,
        "max_tokens": 1000,
        "top_p": 1.0,
        "top_k": 100,
        "stop_sequences": None
    },
    "gemini": {
        "temperature": 0.0,
        "max_output_tokens": 1000,
        "top_p": 1.0,
        "top_k": 40,
        "stop_sequences": None
    },
    "openrouter": {
        "temperature": 0.0,
        "max_tokens": 1000,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stop": None
    }
}

# Parameter descriptions for help text
PARAM_DESCRIPTIONS = {
    "temperature": "Controls randomness: 0.0 is deterministic, higher values increase randomness",
    "max_tokens": "Maximum number of tokens to generate",
    "top_p": "Nucleus sampling: 1.0 considers all tokens, lower values restrict to more likely tokens",
    "top_k": "Only sample from the top K options for each next token",
    "frequency_penalty": "Penalizes repeated tokens: 0.0 is no penalty, higher values discourage repetition",
    "presence_penalty": "Penalizes tokens already present: 0.0 is no penalty, higher values encourage diversity",
    "stop": "Sequences where the API will stop generating further tokens"
}

# Default number of traces for CoT with Self Consistency
DEFAULT_NUM_TRACES = 3

# Default output directory for results
DEFAULT_OUTPUT_DIR = "results"

# Default dataset path
DEFAULT_DATASET_PATH = "dataset.json"

# Default delay between API calls (in seconds)
DEFAULT_SLEEP_TIME = 1.0

# Default retry parameters
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 1.0
DEFAULT_BACKOFF_FACTOR = 2.0

# Default behavior for stopping on errors
DEFAULT_STOP_ON_ERROR = False
