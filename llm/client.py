import json
import re
import asyncio
from openai import AsyncOpenAI
from config.settings import LLM_API_BASE, LLM_API_KEY, LLM_MODEL_NAME, LLM_MAX_TOKENS, LLM_TIMEOUT, SAFE_WORD_BUDGET

class LocalLLM:
    """Hardened LLM client optimized for local inference with Gemma 4 E4B.
    
    Features:
    - Task-specific temperature presets
    - Configurable max_tokens to prevent runaway generation
    - Retry logic for JSON parsing failures
    - Timeout handling for slow/overloaded servers
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=LLM_API_BASE,
            api_key=LLM_API_KEY,
            timeout=LLM_TIMEOUT
        )
        self.model = LLM_MODEL_NAME
        self.default_max_tokens = LLM_MAX_TOKENS
        
        # Diagnostic logging to verify .env loading
        print(f"[LLM] Client initialized. Model: {self.model} | Global Timeout: {LLM_TIMEOUT}s")
        # Running token counters for the current bot session
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
    
    def _log_usage(self, response, label: str = ""):
        """Extracts and logs token usage from an OpenAI-compatible API response."""
        usage = getattr(response, 'usage', None)
        if usage:
            prompt_tok = getattr(usage, 'prompt_tokens', 0) or 0
            completion_tok = getattr(usage, 'completion_tokens', 0) or 0
            self._total_prompt_tokens += prompt_tok
            self._total_completion_tokens += completion_tok
            total_session = self._total_prompt_tokens + self._total_completion_tokens
            print(
                f"[LLM] {label}Tokens: {prompt_tok:,} in → {completion_tok:,} out "
                f"| Session total: {total_session:,}"
            )
            
    def _clean_thinking(self, text: str) -> str:
        """Strips reasoning blocks like <think>...</think> often found in DeepSeek-R1 responses."""
        if not text:
            return ""
        # Remove anything between <think> and </think> tags, case-insensitively and across multiple lines
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()

    async def generate_json(self, system_prompt: str, user_prompt: str, max_retries: int = 2) -> dict:
        """Forces the local LLM to output a dictionary/JSON with retry logic.
        
        On first failure, retries with temperature=0.0 for maximum determinism.
        """
        temperature = 0.1  # Low temperature for strict formatting
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    temperature = 0.0  # Force deterministic on retry
                    print(f"[LLM] Retrying JSON generation (attempt {attempt + 1}/{max_retries}, temp=0.0)...")
                
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=temperature,
                        max_tokens=self.default_max_tokens
                    ),
                    timeout=LLM_TIMEOUT
                )
                
                content = response.choices[0].message.content
                self._log_usage(response, "JSON ")
                
                # First, strip reasoning tags (<think>...</think>) for models like DeepSeek-R1
                content = self._clean_thinking(content)
                
                # Local reasoning models often wrap JSON in markdown blocks or add extra text.
                # This regex surgically extracts just the JSON object/array.
                json_match = re.search(r'\{.*\}|\[.*\]', content, re.DOTALL)
                if json_match:
                    clean_json = json_match.group(0)
                    return json.loads(clean_json)
                else:
                    raise ValueError(f"No JSON found in response: {content[:200]}")
                    
            except asyncio.TimeoutError:
                print(f"[LLM] Timeout after {LLM_TIMEOUT}s on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    continue
                print("[LLM] All retries exhausted due to timeout.")
                return {}
            except json.JSONDecodeError as e:
                print(f"[LLM] JSON parse error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    continue
                print("[LLM] All retries exhausted. Returning empty dict.")
                return {}
            except Exception as e:
                print(f"[LLM] Error generating JSON: {e}")
                return {}

    async def generate_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = None, timeout_override: int = None) -> str:
        """Standard text generation with configurable parameters.
        
        Args:
            system_prompt: System-level instructions
            user_prompt: User query/context
            temperature: Controls randomness (0.0-1.0). Recommended:
                - 0.1: JSON/structured output
                - 0.3: Factual summarization
                - 0.5: Analysis/evaluation
                - 0.7: General conversation
            max_tokens: Max output tokens. Defaults to LLM_MAX_TOKENS setting.
            timeout_override: Override the default global LLM_TIMEOUT in seconds.
        """
        try:
            timeout_val = timeout_override or LLM_TIMEOUT
            
            # Simple heuristic check for large inputs
            total_words = len(system_prompt.split()) + len(user_prompt.split())
            if total_words > 40000:
                print(f"[LLM] ⚠️ Sending massive payload ({total_words:,} words). Inference may be slow.")

            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens or self.default_max_tokens
                ),
                timeout=timeout_val
            )
            self._log_usage(response, "Text ")
            return self._clean_thinking(response.choices[0].message.content)
        except asyncio.TimeoutError:
            print(f"[LLM] Timeout after {timeout_val}s during text generation.")
            return ""
        except Exception as e:
            print(f"[LLM] Error generating text: {e}")
            return ""

    async def generate_text_with_budget(self, system_prompt: str, user_prompt: str, 
                                         max_input_words: int = None, **kwargs) -> str:
        """Text generation with automatic input truncation.
        
        Useful when feeding large scraped content that might exceed context limits.
        Truncates user_prompt to SAFE_WORD_BUDGET to stay within hardware limits.
        """
        budget = max_input_words or SAFE_WORD_BUDGET
        words = user_prompt.split()
        if len(words) > budget:
            print(f"[LLM] Truncating input from {len(words)} to {budget} words")
            user_prompt = " ".join(words[:budget]) + "\n\n[... content truncated for context budget]"
        
        return await self.generate_text(system_prompt, user_prompt, **kwargs)
