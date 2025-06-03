import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
from server.combined.combined import combined_action_execution

app = FastAPI()

# Directory to store credentials
credentials_dir = 'credentials'
os.makedirs(credentials_dir, exist_ok=True)

# Define available tools and context sources
available_tools = ["gmail", "gcalendar", "gdrive", "gdocs", "gslides", "gsheet", "get_short_term_memories", "get_long_term_memories", "web_search"]
context_sources = ["short_term_memories (short-term information about the user)", "long_term_memories (long-term information about the user)", "gdrive (user's Google Drive)"]

class AuthData(BaseModel):
    user_id: str
    token_data: dict

class ExecuteData(BaseModel):
    user_id: str
    action_item: str

@app.post("/auth/google")
async def auth_google(data: AuthData):
    user_dir = os.path.join(credentials_dir, data.user_id)
    os.makedirs(user_dir, exist_ok=True)
    token_file = os.path.join(user_dir, 'google.json')
    with open(token_file, 'w') as f:
        json.dump(data.token_data, f)
    return {"message": "Token saved successfully"}

@app.post("/execute")
async def execute_action(data: ExecuteData):
    token_file = os.path.join(credentials_dir, data.user_id, 'google.json')
    if not os.path.exists(token_file):
        raise HTTPException(status_code=400, detail="Authentication required for Google tools")
    result = combined_action_execution([data.action_item], available_tools, context_sources, data.user_id)
    return result

uvicorn_config = {
    "host": "0.0.0.0",
    "port": 8000,
    "log_level": "info",
}

if __name__ == "__main__":
    uvicorn.run(app, **uvicorn_config)