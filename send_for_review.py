import json, requests, pandas as pd
from pathlib import Path

def main():
    df=pd.read_csv("data/golden_200.csv").head(10)
    out=[]
    for _,r in df.iterrows():
        item={"text":r["text"],"intent":"other","escalation":"Auto-handle",
              "reply":"Thanks for contacting support. We will help you with this request.",
              "evidence":"No historical evidence supplied in rapid prototype."}
        try:
            resp=requests.post("http://127.0.0.1:8000/review",json={"item":item},timeout=60)
            out.append({"text":r["text"],"review":resp.json()})
        except Exception as e:
            out.append({"text":r["text"],"error":str(e)})
    Path("results").mkdir(exist_ok=True)
    json.dump(out,open("results/review_results.json","w"),indent=2)
    print("Saved results/review_results.json")

if __name__=="__main__": main()
