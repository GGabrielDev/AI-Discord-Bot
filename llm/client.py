import json
import re
from openai import AsyncOpenAI
from config.settings import LLM_API_BASE, LLM_API_KEY

class LocalLLM:
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=LLM_API_BASE,
            api_key=LLM_API_KEY
        )
        self.model = "local-model"

    async def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Forces the local LLM to output a dictionary/JSON."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2, # Low temperature for strict formatting
            )
            
            content = response.choices[0].message.content
            
            # Local reasoning models often wrap JSON in markdown blocks or add extra text.
            # This regex surgically extracts just the JSON object/array.
            json_match = re.search(r'\{.*\}|\[.*\]', content, re.DOTALL)
            if json_match:
                clean_json = json_match.group(0)
                return json.loads(clean_json)
            else:
                raise ValueError("No JSON found in response.")
                
        except Exception as e:
            print(f"[LLM] Error generating JSON: {e}")
            return {}

    async def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Standard text generation for summaries."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[LLM] Error generating text: {e}")
            return ""
