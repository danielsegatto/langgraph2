import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from graph import create_app

# 1. Setup
load_dotenv()
app_fastapi = FastAPI(title="Linguistic AI Agent API")

llm = ChatGroq(
    temperature=0.6,
    model_name="openai/gpt-oss-120b", 
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_kwargs={"response_format": {"type": "json_object"}} 
)

# Initialize your LangGraph
langgraph_app = create_app(llm)

# 2. Data Models (What the JSON request looks like)
class TranslationRequest(BaseModel):
    text: str

class RefinementRequest(BaseModel):
    selected_sentence: str

# 3. API Endpoints
@app_fastapi.post("/translate")
async def translate_text(request: TranslationRequest):
    try:
        # We call the graph once per request
        result = langgraph_app.invoke({
            "original_text": request.text, 
            "mode": "translate"
        })
        return {"results": result['final_results']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app_fastapi.post("/refine")
async def refine_text(request: RefinementRequest):
    try:
        result = langgraph_app.invoke({
            "refinement_text": request.selected_sentence, 
            "mode": "refine"
        })
        return {"results": result['final_results']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 is essential for cloud environments like Codespaces
    uvicorn.run(app_fastapi, host="0.0.0.0", port=8000)
