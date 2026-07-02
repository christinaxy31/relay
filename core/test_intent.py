from core.intent_recognizer import IntentRecognizer
r = IntentRecognizer()
for m in ["你好", "我要退款", "应用一直崩溃 error", "应用一直崩溃 太差了", "转人工！紧急"]:
    res = r.recognize(m)
    print(f"{m:30} -> {res.intent.value:10} conf={res.confidence:.2f} urg={res.urgency.name}")