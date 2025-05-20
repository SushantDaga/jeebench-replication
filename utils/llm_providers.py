import requests
import json
import time
from typing import Dict, Any, Optional, Callable, TypeVar, Union
import random
from config import DEFAULT_MAX_RETRIES, DEFAULT_INITIAL_DELAY, DEFAULT_BACKOFF_FACTOR

T = TypeVar('T')

def call_with_retry(
    func: Callable[..., T],
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    **kwargs
) -> Union[T, Dict[str, Any]]:
    """
    Call a function with retry logic and exponential backoff
    
    Args:
        func: Function to call
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        backoff_factor: Factor to increase delay on each retry
        **kwargs: Arguments to pass to the function
        
    Returns:
        Result of the function call or error information
    """
    delay = initial_delay
    last_exception = None
    error_info = {}
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                print(f"Retry attempt {attempt}/{max_retries} after {delay:.2f}s delay...")
            
            result = func(**kwargs)
            
            # If this is a retry, add retry information to the result
            if attempt > 0 and isinstance(result, dict):
                result["retry_info"] = {
                    "attempts": attempt + 1,
                    "success": True
                }
            
            return result
            
        except requests.exceptions.Timeout as e:
            last_exception = e
            error_type = "timeout"
            print(f"Timeout error (attempt {attempt+1}/{max_retries+1}): {str(e)}")
        except requests.exceptions.ConnectionError as e:
            last_exception = e
            error_type = "connection"
            print(f"Connection error (attempt {attempt+1}/{max_retries+1}): {str(e)}")
        except Exception as e:
            # For rate limit errors, we want to retry
            error_message = str(e).lower()
            if ("rate limit" in error_message or 
                "too many requests" in error_message or 
                "429" in error_message):
                last_exception = e
                error_type = "rate_limit"
                print(f"Rate limit error (attempt {attempt+1}/{max_retries+1}): {str(e)}")
            else:
                # For other API errors, don't retry
                error_info = {
                    "error_type": "api_error",
                    "error_message": str(e),
                    "attempts": attempt + 1,
                    "success": False
                }
                return {
                    "error": str(e),
                    "error_info": error_info,
                    "success": False
                }
        
        # If this is the last attempt, prepare error info
        if attempt == max_retries:
            error_info = {
                "error_type": error_type,
                "error_message": str(last_exception),
                "attempts": attempt + 1,
                "success": False
            }
            break
        
        # Add some jitter to the delay to prevent synchronized retries
        jitter = random.uniform(0.8, 1.2)
        actual_delay = delay * jitter
        time.sleep(actual_delay)
        delay *= backoff_factor
    
    # If we've exhausted all retries, return error information
    return {
        "error": str(last_exception),
        "error_info": error_info,
        "success": False
    }

class LLMProvider:
    """Base class for LLM providers"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response from the LLM"""
        raise NotImplementedError("Subclasses must implement this method")

class OpenAIProvider(LLMProvider):
    def generate(
        self, 
        prompt: str, 
        model: str = "gpt-4", 
        temperature: float = 0.0, 
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_delay: float = DEFAULT_INITIAL_DELAY,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call OpenAI API with retry logic
        
        Args:
            prompt: The prompt to send to the model
            model: The model to use
            temperature: Temperature parameter for generation
            max_retries: Maximum number of retries
            initial_delay: Initial delay in seconds
            backoff_factor: Factor to increase delay on each retry
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dictionary containing the response and metadata
        """
        def _call_api():
            import openai
            openai.api_key = self.api_key
            
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                **kwargs
            )
            
            # Extract finish reason if available
            finish_reason = None
            if hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'finish_reason'):
                    finish_reason = response.choices[0].finish_reason
            
            return {
                "provider": "openai",
                "model": model,
                "response": response.choices[0].message.content,
                "raw_response": response,
                "finish_reason": finish_reason,
                "success": True
            }
        
        # Call the API with retry logic
        result = call_with_retry(
            _call_api,
            max_retries=max_retries,
            initial_delay=initial_delay,
            backoff_factor=backoff_factor
        )
        
        # If the result indicates an error, format it properly
        if not result.get("success", True):
            error_msg = result.get("error", "Unknown error")
            print(f"Error calling OpenAI API after retries: {error_msg}")
            return {
                "provider": "openai",
                "model": model,
                "response": f"Error: {error_msg}",
                "raw_response": None,
                "error": error_msg,
                "error_info": result.get("error_info", {}),
                "success": False
            }
        
        return result

class AnthropicProvider(LLMProvider):
    def generate(
        self, 
        prompt: str, 
        model: str = "claude-3-opus-20240229", 
        temperature: float = 0.0, 
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_delay: float = DEFAULT_INITIAL_DELAY,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call Anthropic API with retry logic
        
        Args:
            prompt: The prompt to send to the model
            model: The model to use
            temperature: Temperature parameter for generation
            max_retries: Maximum number of retries
            initial_delay: Initial delay in seconds
            backoff_factor: Factor to increase delay on each retry
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dictionary containing the response and metadata
        """
        def _call_api():
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            
            response = client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                **kwargs
            )
            
            # Extract stop reason if available
            stop_reason = None
            if hasattr(response, 'stop_reason'):
                stop_reason = response.stop_reason
            
            return {
                "provider": "anthropic",
                "model": model,
                "response": response.content[0].text,
                "raw_response": response,
                "finish_reason": stop_reason,
                "success": True
            }
        
        # Call the API with retry logic
        result = call_with_retry(
            _call_api,
            max_retries=max_retries,
            initial_delay=initial_delay,
            backoff_factor=backoff_factor
        )
        
        # If the result indicates an error, format it properly
        if not result.get("success", True):
            error_msg = result.get("error", "Unknown error")
            print(f"Error calling Anthropic API after retries: {error_msg}")
            return {
                "provider": "anthropic",
                "model": model,
                "response": f"Error: {error_msg}",
                "raw_response": None,
                "error": error_msg,
                "error_info": result.get("error_info", {}),
                "success": False
            }
        
        return result

class GeminiProvider(LLMProvider):
    def generate(
        self, 
        prompt: str, 
        model: str = "gemini-1.5-pro", 
        temperature: float = 0.0, 
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_delay: float = DEFAULT_INITIAL_DELAY,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call Google Gemini API with retry logic
        
        Args:
            prompt: The prompt to send to the model
            model: The model to use
            temperature: Temperature parameter for generation
            max_retries: Maximum number of retries
            initial_delay: Initial delay in seconds
            backoff_factor: Factor to increase delay on each retry
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dictionary containing the response and metadata
        """
        def _call_api():
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            model_obj = genai.GenerativeModel(model_name=model)
            
            generation_config = {"temperature": temperature}
            for key, value in kwargs.items():
                generation_config[key] = value
                
            response = model_obj.generate_content(prompt, generation_config=generation_config)
            
            # Extract finish reason if available
            finish_reason = None
            if hasattr(response, 'candidates') and len(response.candidates) > 0:
                if hasattr(response.candidates[0], 'finish_reason'):
                    finish_reason = response.candidates[0].finish_reason
            
            return {
                "provider": "gemini",
                "model": model,
                "response": response.text,
                "raw_response": response,
                "finish_reason": finish_reason,
                "success": True
            }
        
        # Call the API with retry logic
        result = call_with_retry(
            _call_api,
            max_retries=max_retries,
            initial_delay=initial_delay,
            backoff_factor=backoff_factor
        )
        
        # If the result indicates an error, format it properly
        if not result.get("success", True):
            error_msg = result.get("error", "Unknown error")
            print(f"Error calling Gemini API after retries: {error_msg}")
            return {
                "provider": "gemini",
                "model": model,
                "response": f"Error: {error_msg}",
                "raw_response": None,
                "error": error_msg,
                "error_info": result.get("error_info", {}),
                "success": False
            }
        
        return result

class OpenRouterProvider(LLMProvider):
    def generate(
        self, 
        prompt: str, 
        model: str = "openai/gpt-4", 
        temperature: float = 0.0, 
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_delay: float = DEFAULT_INITIAL_DELAY,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call OpenRouter API with retry logic
        
        Args:
            prompt: The prompt to send to the model
            model: The model to use
            temperature: Temperature parameter for generation
            max_retries: Maximum number of retries
            initial_delay: Initial delay in seconds
            backoff_factor: Factor to increase delay on each retry
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dictionary containing the response and metadata
        """
        def _call_api():
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature
            }
            
            # Add any additional parameters
            for key, value in kwargs.items():
                if key not in data:
                    data[key] = value
            
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60  # Add timeout to catch connection issues
            )
            
            response_json = response.json()
            
            if response.status_code != 200:
                raise Exception(f"OpenRouter API returned status code {response.status_code}: {response_json}")
            
            # Extract finish reason if available
            finish_reason = None
            if "choices" in response_json and len(response_json["choices"]) > 0:
                if "finish_reason" in response_json["choices"][0]:
                    finish_reason = response_json["choices"][0]["finish_reason"]
            
            return {
                "provider": "openrouter",
                "model": model,
                "response": response_json["choices"][0]["message"]["content"],
                "raw_response": response_json,
                "finish_reason": finish_reason,
                "success": True
            }
        
        # Call the API with retry logic
        result = call_with_retry(
            _call_api,
            max_retries=max_retries,
            initial_delay=initial_delay,
            backoff_factor=backoff_factor
        )
        
        # If the result indicates an error, format it properly
        if not result.get("success", True):
            error_msg = result.get("error", "Unknown error")
            print(f"Error calling OpenRouter API after retries: {error_msg}")
            return {
                "provider": "openrouter",
                "model": model,
                "response": f"Error: {error_msg}",
                "raw_response": None,
                "error": error_msg,
                "error_info": result.get("error_info", {}),
                "success": False
            }
        
        return result

def get_provider(provider_name: str, api_key: str) -> LLMProvider:
    """
    Factory function to get the appropriate provider
    
    Args:
        provider_name: Name of the provider
        api_key: API key for the provider
        
    Returns:
        LLMProvider instance
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "openrouter": OpenRouterProvider
    }
    
    if provider_name not in providers:
        raise ValueError(f"Provider {provider_name} not supported")
    
    return providers[provider_name](api_key)
