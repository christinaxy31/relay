import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class IntentCategory(Enum):
    QUERY      = "query"
    COMPLAINT  = "complaint"
    REQUEST    = "request"
    GREETING   = "greeting"
    ESCALATION = "escalation"
    TECHNICAL  = "technical"
    BILLING    = "billing"
    ACCOUNT    = "account"
    FEEDBACK   = "feedback"
    OTHER      = "other"


class UrgencyLevel(Enum):
    LOW = 1; MEDIUM = 2; HIGH = 3; CRITICAL = 4


@dataclass
class IntentResult:
    intent:     IntentCategory
    confidence: float
    urgency:    UrgencyLevel
    reasoning:  str = ""
    latency_ms: float = 0.0


_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ["紧急", "emergency", "urgent", "asap", "立刻"],
    UrgencyLevel.HIGH:     ["今天", "马上", "尽快", "now"],
    UrgencyLevel.MEDIUM:   ["这周", "soon", "快点"],
}


class IntentRecognizer:
    def __init__(self, confidence_threshold: float = 0.5):
        self.threshold = confidence_threshold

    def recognize(self, message: str) -> IntentResult:
        t0 = time.monotonic()
        pat = self._pattern_recognize(message)
        intent = pat["intent"] if pat["confidence"] >= self.threshold else IntentCategory.OTHER
        return IntentResult(
            intent=intent,
            confidence=pat["confidence"],
            urgency=self._urgency(message, intent),
            reasoning="pattern-only",
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        msg = message.lower()
        patterns = {
            IntentCategory.ESCALATION: ["投诉", "经理", "转人工", "supervisor"],
            IntentCategory.COMPLAINT:  ["太差", "糟糕", "horrible"],
            IntentCategory.QUERY:      ["?", "？", "怎么", "什么", "status"],
            IntentCategory.REQUEST:    ["帮我", "需要", "please", "help"],
            IntentCategory.GREETING:   ["你好", "嗨", "hello", "hi"],
            IntentCategory.BILLING:    ["退款", "扣款", "发票", "refund"],
            IntentCategory.TECHNICAL:  ["崩溃", "报错", "error", "crash"],
            IntentCategory.ACCOUNT:    ["密码", "邮箱", "账户", "password"],
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