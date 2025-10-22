import os

API_MAPPINGS = [
    ("ALPHA_VANTAGE_API_KEY", "alpha_vantage_api_key"),
    ("FRED_API_KEY", "fred_api_key"),
    ("NEWS_API_KEY", "news_api_key"),
    ("MODEL_NAME", "groq_model_name"),
    ("GROQ_API_KEY", "groq_api_key"),
    ("TEMPERATURE", "temperature"),
    ("TWITTER_BEARER_TOKEN", "twitter_bearer_token"),
    ("API_RATE_LIMIT_DELAY", "api_rate_limit_delay"),
    ("REQUEST_TIMEOUT", "request_timeout"),
    ("GROQ_TOOL_CHOICE", "groq_tool_choice"),
    ("MAX_TOOL_CALLS", "max_tool_calls"),
    ("DISCLAIMER_TEXT", "disclaimer_text"),
]

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "moonshotai/kimi-k2-instruct-0905")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
API_RATE_LIMIT_DELAY = 60 / 5
REQUEST_TIMEOUT = 10
GROQ_TOOL_CHOICE = os.getenv("GROQ_TOOL_CHOICE", "auto")
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "5"))
DISCLAIMER_TEXT = os.getenv("DISCLAIMER_TEXT", "This is AI-generated analysis for informational purposes only. Not financial advice. Consult a professional. Data as of latest fetch.")