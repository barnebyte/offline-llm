from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from run_llm import generate_response

app = FastAPI()

# Make sure CORS middleware is added before any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],  # This allows POST, OPTIONS, etc.
    allow_headers=["*"],  # This ensures Content-Type and other headers are allowed
)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(prompt_request: PromptRequest):
    prompt = prompt_request.prompt
    response = generate_response(prompt)
    return {"response": response}