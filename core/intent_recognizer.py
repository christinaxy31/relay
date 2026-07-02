import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


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

    async def recognize(self, message: str) -> IntentResult:
        t0 = time.monotonic()
        pattern_result = self._pattern_recognize(message)

        if self.client is not None:
            llm_result = await self._llm_recognize(message)
            chosen = llm_result if not llm_result.get("failed") else pattern_result
            if chosen["confidence"] >= self.threshold:
                intent = chosen["intent"]
                confidence = chosen["confidence"]
                reasoning = chosen.get("reasoning", "")
            else:
                intent = pattern_result["intent"]
                confidence = pattern_result["confidence"]
                reasoning = "pattern-only"
        else:
            intent = pattern_result["intent"] if pattern_result["confidence"] >= self.threshold else IntentCategory.OTHER
            confidence = pattern_result["confidence"]
            reasoning = "pattern-only"

        return IntentResult(
            intent=intent,
            confidence=confidence,
            urgency=self._urgency(message, intent),
            reasoning=reasoning,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

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