from smolagents import HfApiModel, LiteLLMModel

DEFAULTS = {
    "provider": "ollama",
    "model_id": "qwen2.5-coder:7b",
    "api_base": "http://localhost:11434",
    "api_key": None,
}

def load_model(
    provider: str = None,
    model_id: str = None,
    api_base: str = None,
    api_key: str = None,
):
    """
    Load a model object from a provider. 
    """
    # Load defaults if None
    if provider is None:
        provider = DEFAULTS["provider"]
    if model_id is None:
        model_id = DEFAULTS["model_id"]
    if api_base is None:
        api_base = DEFAULTS["api_base"]
    if api_key is None:
        api_key = DEFAULTS["api_key"]


    if provider == "huggingface":
        model = HfApiModel(model_id, token=api_key)
    elif provider == "ollama":
        model = LiteLLMModel(f"{provider}/{model_id}", api_base=api_base, api_key=api_key)
    elif provider == "openai":
        model = LiteLLMModel(model_id, api_base=api_base, api_key=api_key)
    else:
        raise ValueError(f"Provider {provider} not recognized.")
    
    return model