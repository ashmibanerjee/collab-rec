import uvicorn
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from .endpoints import router
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[1] / "adk"/ ".config" / ".env"
load_dotenv(dotenv_path=env_path)
app = FastAPI(title="Collab-REC ADK Backend API")
app.include_router(router)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
