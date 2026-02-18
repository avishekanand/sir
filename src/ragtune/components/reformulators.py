from typing import List, Optional
from ragtune.core.interfaces import BaseReformulator
from ragtune.core.types import RAGtuneContext
from ragtune.registry import registry
from ragtune.utils.config import config

@registry.reformulator("identity")
class IdentityReformulator(BaseReformulator):
    """Pass-through reformulator that returns the original query."""
    def generate(self, context: RAGtuneContext) -> List[str]:
        if context.tracker.try_consume_reformulation():
            return [context.query]
        return []

@registry.reformulator("llm_rewrite")
class LLMReformulator(BaseReformulator):
    """LLM-based query rewriting with robust parsing and filtering."""
    def __init__(self, model_name: str = "gpt-4o-mini", api_base: Optional[str] = None):
        self.model = model_name
        self.api_base = api_base

    def generate(self, context: RAGtuneContext) -> List[str]:
        if not context.tracker.try_consume_reformulation():
            return []
            
        import litellm
        
        try:
            m = config.get("retrieval.num_reformulations", 2)
            max_tokens = config.get("retrieval.max_reformulation_tokens", 1000)
            prompt_config = config.get_prompt("reformulation.llm_rewrite")
            
            if not prompt_config:
                system_prompt = "You are a search expert."
                user_prompt = f"Rewrite query '{context.query}' into {m} variations."
            else:
                system_prompt = prompt_config.get("system", "You are a search expert.")
                user_prompt = prompt_config.get("user", "").format(query=context.query, m=m)
            
            response = litellm.completion(
                model=self.model,
                api_base=self.api_base,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            queries = self._parse_response(content)
            return self._filter_queries(queries, context.query, m)
        except Exception:
            return []

    async def agenerate(self, context: RAGtuneContext) -> List[str]:
        if not context.tracker.try_consume_reformulation():
            return []
            
        import litellm
        
        m = config.get("retrieval.num_reformulations", 2)
        max_tokens = config.get("retrieval.max_reformulation_tokens", 1000)
        prompt_config = config.get_prompt("reformulation.llm_rewrite")
        
        system_prompt = prompt_config.get("system")
        user_prompt = prompt_config.get("user").format(query=context.query, m=m)
        
        try:
            response = await litellm.acompletion(
                model=self.model,
                api_base=self.api_base,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            queries = self._parse_response(content)
            return self._filter_queries(queries, context.query, m)
        except Exception:
            return []

    def _parse_response(self, content: str) -> List[str]:
        import json
        import re
        
        # 1. Strip code fences
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*", "", content)
        content = content.strip()
        
        # 2. Try to find a JSON-like structure [...] if it's wrapped in text
        if not content.startswith("[") and "[" in content:
            match = re.search(r"(\[.*\])", content, re.DOTALL)
            if match:
                content = match.group(1)
        elif not content.startswith("{") and "{" in content:
            match = re.search(r"(\{.*\})", content, re.DOTALL)
            if match:
                content = match.group(1)
        
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return [str(q) for q in data if q]
            elif isinstance(data, dict):
                # Try common keys
                reformulations = data.get("reformulations") or data.get("queries")
                if not reformulations:
                    # Look for any list in the dict
                    for val in data.values():
                        if isinstance(val, list):
                            reformulations = val
                            break
                
                if isinstance(reformulations, list):
                    return [str(q) for q in reformulations if q]
                elif not reformulations and data:
                     # Maybe it's just a single KV pair?
                     return [str(v) for v in data.values() if isinstance(v, str)]
        except (json.JSONDecodeError, TypeError, IndexError):
            pass
        return []

    def _filter_queries(self, queries: List[str], original_query: str, m: int) -> List[str]:
        from difflib import SequenceMatcher
        
        threshold = config.get("retrieval.near_duplicate_threshold", 0.8)
        filtered = []
        original_norm = original_query.lower().strip()
        
        for q in queries:
            q_clean = q.strip()
            if not q_clean:
                continue
            
            q_norm = q_clean.lower()
            
            # Drop original query
            if q_norm == original_norm:
                continue
            
            # Near-duplicate filtering
            is_duplicate = False
            for existing in filtered:
                if SequenceMatcher(None, q_norm, existing.lower()).ratio() > threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(q_clean)
                
            if len(filtered) >= m:
                break
                
        return filtered
