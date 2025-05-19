import requests
import json
import time
from typing import Dict, Any, Optional

class LLMProvider:
    """Base class for LLM providers"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a response from the LLM"""
        raise NotImplementedError("Subclasses must implement this method")

class OpenAIProvider(LLMProvider):
    def generate(self, prompt: str, model: str = "gpt-4", temperature: float = 0.0, **kwargs) -> Dict[str, Any]:
        """
        Call OpenAI API
        
        Args:
            prompt: The prompt to send to the model
            model: The model to use
            temperature: Temperature parameter for generation
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
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
                "finish_reason": finish_reason
            }
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            return {
                "provider": "openai",
                "model": model,
                "response": f"Error: {str(e)}",
                "raw_response": None,
                "error": str(e)
            }

class AnthropicProvider(LLMProvider):
    def generate(self, prompt: str, model: str = "claude-3-opus-20240229", temperature: float = 0.0, **kwargs) -> Dict[str, Any]:
        """
        Call Anthropic API
        
        Args:
            prompt: The prompt to send to the model
            model: The model to use
            temperature: Temperature parameter for generation
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
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
                "finish_reason": stop_reason
            }
        except Exception as e:
            print(f"Error calling Anthropic API: {str(e)}")
            return {
                "provider": "anthropic",
                "model": model,
                "response": f"Error: {str(e)}",
                "raw_response": None,
                "error": str(e)
            }

class GeminiProvider(LLMProvider):
    def generate(self, prompt: str, model: str = "gemini-1.5-pro", temperature: float = 0.0, **kwargs) -> Dict[str, Any]:
        """
        Call Google Gemini API
        
        Args:
            prompt: The prompt to send to the model
            model: The model to use
            temperature: Temperature parameter for generation
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
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
                "finish_reason": finish_reason
            }
        except Exception as e:
            print(f"Error calling Gemini API: {str(e)}")
            return {
                "provider": "gemini",
                "model": model,
                "response": f"Error: {str(e)}",
                "raw_response": None,
                "error": str(e)
            }

class OpenRouterProvider(LLMProvider):
    def generate(self, prompt: str, model: str = "openai/gpt-4", temperature: float = 0.0, **kwargs) -> Dict[str, Any]:
        """
        Call OpenRouter API
        
        Args:
            prompt: The prompt to send to the model
            model: The model to use
            temperature: Temperature parameter for generation
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
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
                json=data
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
                "finish_reason": finish_reason
            }
        except Exception as e:
            print(f"Error calling OpenRouter API: {str(e)}")
            return {
                "provider": "openrouter",
                "model": model,
                "response": f"Error: {str(e)}",
                "raw_response": None,
                "error": str(e)
            }

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
