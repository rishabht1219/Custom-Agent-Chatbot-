from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from src.Agent import CustomAgent

agent  = CustomAgent()

app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/chat")
async def chat(query: Query):
    if query.query:
        # Process the query here. For example, return a simple echo of the query.
        response = agent.chat(query.query)
        return {"response": response}
    else:
        # If query is empty or invalid, raise an HTTP exception.
        raise HTTPException(status_code=400, detail="No valid query provided.")

# #To run the server:
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8001)
