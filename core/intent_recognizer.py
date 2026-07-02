import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from typing import List, Any, Dict, Optional
from anthropic import AsyncAnthropic
import asyncio, os
import hashlib


logger = logging.getLogger(__name__)


def _cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = sum(x*x for x in a) ** 0.5
    nb = sum(x*x for x in b) ** 0.5
    return dot / (na*nb) if na and nb else 0.0




class IntentCategory(Enum):
    QUERY = "query"
    COMPLAINT = "complaint"
    REQUEST = "request"
    GREETING = "greeting"
    ESCALATION = "escalation"
    TECHNICAL = "technical"
    BILLING = "billing"
    ACCOUNT = "account"
    FEEDBACK = "feedback"
    OTHER = "other"


class UrgencyLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent: IntentCategory
    confidence: float
    urgency: UrgencyLevel
    reasoning: str = ""
    latency_ms: float = 0.0
    votes: Dict[str, Any] = field(default_factory=dict) 

_TEMPLATES = {
    IntentCategory.QUERY: ["What's my order status?", "How do I reset my password?"],
    IntentCategory.COMPLAINT: ["Your service is terrible!", "No one has helped me!"],
    IntentCategory.REQUEST: ["Help me cancel my order", "Please help me get a refund"],
    IntentCategory.GREETING: ["Hello", "Good morning"],
    IntentCategory.ESCALATION: ["Transfer me to a human agent", "I want to speak to your manager"],
    IntentCategory.TECHNICAL: ["The app keeps crashing", "I'm getting a 500 error"],
    IntentCategory.BILLING: ["Why was I charged twice?", "I want to request a refund"],
    IntentCategory.ACCOUNT: ["Change my email address", "Delete my account"],
    IntentCategory.FEEDBACK: ["Great service!", "Very satisfied"],
}

_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ["emergency", "urgent", "asap", "immediately", "right now"],
    UrgencyLevel.HIGH: ["today", "now", "quickly", "hurry"],
    UrgencyLevel.MEDIUM: ["this week", "soon"],
}


class IntentRecognizer:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        confidence_threshold: float = 0.5,
    ):
        self.threshold = confidence_threshold
        self.model = model
        self.client = None

        if api_key:
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self.client = AsyncAnthropic(**kwargs)
        
        self._embedding_enabled = not bool(base_url)   # 3rd-party APIs lack embeddings
        self._tpl_embeddings: Dict[IntentCategory, List[List[float]]] = {}
        self._cache = {}
        self.cache_hits = self.cache_misses = 0


    
    async def recognize(self, message: str) -> IntentResult:
        key = message.strip()[:200]
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        # Pattern layer is synchronous (pure CPU, zero-latency) -> compute directly.
        pat = self._pattern_recognize(message)

        # LLM and embedding are I/O-bound -> launch as tasks so they run concurrently.
        llm_task = asyncio.create_task(self._llm_recognize(message)) if self.client is not None else None
        emb_task = asyncio.create_task(self._embedding_recognize(message)) if self._embedding_enabled else None

        if llm_task and emb_task:
            llm, emb = await asyncio.gather(llm_task, emb_task)
        elif llm_task:
            llm = await llm_task
            emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}
        elif emb_task:
            emb = await emb_task
            llm = {"intent": IntentCategory.OTHER, "confidence": 0.0, "failed": True}
        else:  # pattern-only (no client, embedding disabled)
            llm = {"intent": IntentCategory.OTHER, "confidence": 0.0, "failed": True}
            emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}

        intent = self._vote(llm, emb, pat)
        result = IntentResult(
            intent=intent,
            confidence=llm.get("confidence", 0.0),
            urgency=self._urgency(message, intent),
            reasoning=llm.get("reasoning", "") or "pattern/embedding",
            latency_ms=(time.monotonic() - t0) * 1000,
            votes={
                "llm": {"intent": llm["intent"].value, "conf": round(llm.get("confidence", 0.0), 2)},
                "emb": {"intent": emb["intent"].value, "conf": round(emb.get("confidence", 0.0), 2)},
                "pat": {"intent": pat["intent"].value, "conf": round(pat.get("confidence", 0.0), 2)},
            },
        )

        # Write to cache (simple size cap).
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
        self._cache[key] = result
        return result


    async def _llm_recognize(self, message: str) -> Dict[str, Any]:
        examples = "\n".join(f'  "{tpls[0]}" -> {cat.value}' for cat, tpls in _TEMPLATES.items())
        prompt = f"""You are a customer-service intent classification expert.
Based on the examples, determine the user's intent and return JSON.
Examples:
{examples}

User message: "{message}"
Return format (JSON only): {{"intent":"<value>","confidence":<0-1>,"reasoning":"<one sentence>"}}
Available intents: {', '.join(c.value for c in IntentCategory)}"""
        try:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
            data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            try:
                data["intent"] = IntentCategory(data["intent"])
            except ValueError:
                data["intent"] = IntentCategory.OTHER
            return data
        except Exception as ex:
            logger.warning(f"LLM recognition failed: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "failed": True}


    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        msg = message.lower()
        patterns = {
            IntentCategory.ESCALATION: ["complaint", "manager", "human agent", "supervisor"],
            IntentCategory.COMPLAINT: ["terrible", "awful", "horrible", "worst"],
            IntentCategory.QUERY: ["?", "how", "what", "status", "when"],
            IntentCategory.REQUEST: ["help me", "i need", "please", "can you"],
            IntentCategory.GREETING: ["hello", "hi", "hey", "good morning"],
            IntentCategory.BILLING: ["refund", "charge", "invoice", "billing", "payment"],
            IntentCategory.TECHNICAL: ["crash", "error", "bug", "not working", "500", "401"],
            IntentCategory.ACCOUNT: ["password", "email", "account", "login"],
        }
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if kw in msg)
            if hits:
                score = hits / len(kws)
                if score > best_score:
                    best_score, best_cat = score, cat
        return {"intent": best_cat, "confidence": best_score}



    async def _load_template_embeddings(self):
        for cat, tpls in _TEMPLATES.items():
            if cat not in self._tpl_embeddings:
                self._tpl_embeddings[cat] = [await self._embed_text(t) for t in tpls]

    async def _embed_text(self, text: str):
        embeddings = getattr(self.client, "embeddings", None)   # None-safe: works even if client is None
        if embeddings is not None:
            try:
                resp = await embeddings.create(model="voyage-3-lite", input=[text])
                return list(resp.data[0].embedding)
            except Exception as ex:
                logger.warning(f"Remote embedding failed, using local fallback: {ex}")
        return self._local_embedding(text)

    async def _embedding_recognize(self, message: str):
        try:
            await self._load_template_embeddings()
            mv = await self._embed_text(message)
            best_cat, best = IntentCategory.OTHER, 0.0
            for cat, vecs in self._tpl_embeddings.items():
                s = max(_cosine(mv, v) for v in vecs)
                if s > best:
                    best, best_cat = s, cat
            return {"intent": best_cat, "confidence": best}
        except Exception as ex:
            logger.warning(f"Embedding recognition failed: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0}

    @staticmethod
    def _local_embedding(text, dims=256):
        import hashlib
        norm = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(norm) >= n:
                tokens.update(norm[i:i+n] for i in range(len(norm)-n+1))
        for tok in (tokens or {norm}):
            d = hashlib.md5(tok.encode()).digest()
            idx = int.from_bytes(d[:4], "big") % dims
            vec[idx] += 1.0 if d[4] % 2 == 0 else -1.0
        return vec





    def _vote(self, llm, emb, pat):
        # If the LLM path is unavailable/failed, fall back to embedding, then pattern.
        if llm.get("failed"):
            for r in (emb, pat):
                if r["intent"] != IntentCategory.OTHER and r["confidence"] > 0:
                    return r["intent"]
            return IntentCategory.OTHER

        # Weighted vote; when embedding is disabled, redistribute its weight.
        if self._embedding_enabled:
            weights = [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
        else:
            weights = [(llm, 0.85), (pat, 0.15)]

        scores: Dict[IntentCategory, float] = {}
        for r, w in weights:
            scores[r["intent"]] = scores.get(r["intent"], 0.0) + w * r["confidence"]
        best = max(scores, key=scores.get)
        return best if scores[best] >= self.threshold else IntentCategory.OTHER
    
    

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        msg = message.lower()
        for level, kws in _URGENCY_KEYWORDS.items():
            if any(kw in msg for kw in kws):
                return level
        if intent == IntentCategory.ESCALATION:
            return UrgencyLevel.HIGH
        if intent == IntentCategory.COMPLAINT:
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW