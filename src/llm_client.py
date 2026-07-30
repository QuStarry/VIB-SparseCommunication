"""
DeepSeek LLM Client for VIB-SparseComm Multi-Agent System

Provides an OpenAI-compatible interface to DeepSeek API for agent reasoning.
"""

import os
import time
from typing import List, Dict, Optional


def resolve_deepseek_api_key(
    api_key: Optional[str] = None,
    fallback_env_path: str = "configs/pro.env"
) -> Optional[str]:
    """
    Resolve DeepSeek API key from:
    1. explicit argument
    2. DEEPSEEK_API_KEY environment variable
    3. local env-style config file, e.g. configs/pro.env
    """
    if api_key:
        return api_key

    env_key = os.environ.get("DEEPSEEK_API_KEY")
    if env_key:
        return env_key

    if not os.path.exists(fallback_env_path):
        return None

    try:
        with open(fallback_env_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() != "DEEPSEEK_API_KEY":
                    continue
                value = value.strip().strip('"').strip("'")
                if value:
                    return value
    except OSError:
        return None

    return None


class DeepSeekClient:
    """
    DeepSeek API Client (OpenAI-compatible interface).
    
    Uses the DeepSeek chat completion API for agent reasoning tasks.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        """
        Initialize DeepSeek client.
        
        Args:
            api_key: DeepSeek API key (reads from DEEPSEEK_API_KEY env if not provided)
            base_url: API base URL
            model: Model name (deepseek-chat or deepseek-reasoner)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = resolve_deepseek_api_key(api_key=api_key)
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key required. Set DEEPSEEK_API_KEY env variable "
                "or pass api_key parameter."
            )
        
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Token usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
        
        # Initialize OpenAI-compatible client
        self._client = None
    
    @property
    def client(self):
        """Lazy initialize the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except ImportError:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                )
        return self._client
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Dict with 'content', 'prompt_tokens', 'completion_tokens'
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )
            
            # Track token usage
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_calls += 1
            
            return {
                "content": response.choices[0].message.content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
            
        except Exception as e:
            print(f"[DeepSeek API Error]: {e}")
            # Retry once after a short delay
            time.sleep(2)
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens
                )
                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                completion_tokens = response.usage.completion_tokens if response.usage else 0
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_calls += 1
                
                return {
                    "content": response.choices[0].message.content,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            except Exception as retry_e:
                print(f"[DeepSeek API Retry Failed]: {retry_e}")
                return {
                    "content": f"[API Error: {retry_e}]",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
    
    def get_token_stats(self) -> Dict:
        """Get cumulative token usage statistics."""
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens
        }
    
    def reset_stats(self):
        """Reset token usage statistics."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0
