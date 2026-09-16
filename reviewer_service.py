import os, re, json
from fastapi import FastAPI
from openai import OpenAI

app=FastAPI(title="Secondary Reviewer")
client=OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) if os.environ.get("OPENAI_API_KEY") else None

RUBRIC=[
"Classification: does the predicted intent match the customer message and codebook?",
"Escalation: is the auto-handle/escalate decision justified by the stated policy?",
"Grounding: is the draft reply supported by the customer message and historical evidence supplied?",
"Format/policy: does the response follow the required output format and avoid invented account-specific facts?"
]

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/review")
def review(payload:dict):
    item=payload["item"]
    escalation_format=bool(re.match(r"^(Auto-handle|Escalate: .+)$", str(item.get("escalation","")), re.I))
    if client is None:
        return {"reviewer_status":"NO_API_KEY","deterministic":{"escalation_format_valid":escalation_format}}
    prompt=f"""You are a strict secondary reviewer auditing a customer-support AI.
Return JSON only with scores 0,1,2 for four criteria and a short note.
Criteria:
1 classification correctness
2 escalation correctness
3 reply grounding
4 format/policy compliance
Customer: {item.get('text','')}
Predicted intent: {item.get('intent','')}
Escalation: {item.get('escalation','')}
Reply: {item.get('reply','')}
Historical evidence: {item.get('evidence','')}
Score meaning: 2=good, 1=partially correct, 0=incorrect.
"""
    r=client.responses.create(model=os.environ.get("JUDGE_MODEL","gpt-5.6-luna"),input=prompt)
    raw=r.output_text.strip()
    try: data=json.loads(raw)
    except: data={"raw":raw}
    data["deterministic_escalation_format_valid"]=escalation_format
    return data

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=int(os.environ.get("REVIEWER_PORT","8000")))
