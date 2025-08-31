"""
Simplified LLM Interface for CryptoTrade
Supports: OpenAI-compatible APIs (OpenAI, vLLM, Ollama) and llama-cpp-python
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import os
from tenacity import retry, stop_after_attempt, wait_random_exponential
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LLMInterface(ABC):
    """Abstract base class for LLM backends"""
    
    @abstractmethod
    def generate(self, prompt: str, model: str, seed: int, 
                temperature: float = 0.0, max_tokens: int = 256) -> str:
        """Generate response from the LLM"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available/initialized"""
        pass


class OpenAILLM(LLMInterface):
    """OpenAI/Compatible API backend (OpenAI, vLLM, Ollama)"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key or os.getenv('OPENAI_API_KEY'),
                base_url=base_url or os.getenv('OPENAI_API_BASE')
            )
            self._available = True
        except ImportError:
            print("OpenAI library not installed. Run: pip install openai")
            self._available = False
        except Exception as e:
            print(f"Failed to initialize OpenAI client: {e}")
            self._available = False
    
    def is_available(self) -> bool:
        return self._available
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate(self, prompt: str, model: str, seed: int, 
                temperature: float = 0.0, max_tokens: int = 256) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed
        )
        return response.choices[0].message.content


class LlamaCppLLM(LLMInterface):
    """llama-cpp-python backend for GGUF models"""
    
    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1, 
                 n_threads: int = -1, verbose: bool = False):
        try:
            from llama_cpp import Llama
            
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,          # Context window
                n_threads=n_threads,  # Use all threads
                n_gpu_layers=n_gpu_layers,  # Use GPU if available
                verbose=verbose
            )
            self._available = True
            print(f"llama-cpp model loaded: {model_path}")
            
        except ImportError:
            print("llama-cpp-python not installed. Run: pip install llama-cpp-python")
            self._available = False
        except Exception as e:
            print(f"Failed to load llama-cpp model: {e}")
            self._available = False
    
    def is_available(self) -> bool:
        return self._available
    
    @retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def generate(self, prompt: str, model: str, seed: int, 
                temperature: float = 0.0, max_tokens: int = 256) -> str:
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            echo=False,  # Don't echo the prompt back
        )
        return output['choices'][0]['text'].strip()


# Global LLM instance
_llm_instance: Optional[LLMInterface] = None


def initialize_openai_llm(api_key: Optional[str] = None, base_url: Optional[str] = None) -> LLMInterface:
    """Initialize OpenAI-compatible LLM (OpenAI, vLLM, Ollama)"""
    global _llm_instance
    _llm_instance = OpenAILLM(api_key=api_key, base_url=base_url)
    
    if not _llm_instance.is_available():
        raise RuntimeError("Failed to initialize OpenAI-compatible backend")
    
    return _llm_instance


def initialize_llamacpp_llm(model_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1, 
                           n_threads: int = -1, verbose: bool = False) -> LLMInterface:
    """Initialize llama-cpp-python LLM"""
    global _llm_instance
    _llm_instance = LlamaCppLLM(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        verbose=verbose
    )
    
    if not _llm_instance.is_available():
        raise RuntimeError("Failed to initialize llama-cpp backend")
    
    return _llm_instance


def get_llm() -> LLMInterface:
    """Get global LLM instance"""
    if _llm_instance is None:
        raise RuntimeError("LLM not initialized. Call initialize_openai_llm() or initialize_llamacpp_llm() first")
    return _llm_instance


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_chat(prompt: str, model: str, seed: int, temperature: float = 0.0, 
             max_tokens: int = 256, stop_strs: Optional[List[str]] = None, 
             is_batched: bool = False, debug: bool = False) -> str:
    """
    Compatible function with original utils.py
    
    This maintains the exact same interface as the original get_chat function
    """
    llm = get_llm()
    
    response = llm.generate(
        prompt=prompt,
        model=model,
        seed=seed,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    if debug:
        print(f"Backend: {type(llm).__name__}")
    
    return response


if __name__ == "__main__":
    # Example: Test with llama-cpp
    try:
        initialize_llamacpp_llm(
            model_path="./models/Qwen2.5-14B-Instruct-Q6_K.gguf",
            n_ctx=2048,
            n_gpu_layers=48,
            verbose=True
        )
        response = get_chat("Hello, how are you?", "llama", 42, debug=True)
        print("Response:", response)
    except Exception as e:
        print(f"Error: {e}")