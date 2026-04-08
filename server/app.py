from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from .env import CodeDebugEnv, DebugAction
import uuid

app = FastAPI(title="Code Debug RL Environment")
env = CodeDebugEnv()


class StepRequest(BaseModel):
    fixed_code: str
    use_hint: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(bug_id: Optional[str] = None):
    sid = str(uuid.uuid4())
    obs = env.reset(session_id=sid, bug_id=bug_id)
    return {"session_id": sid, "observation": obs.model_dump()}


@app.post("/step/{session_id}")
def step(session_id: str, request: StepRequest):
    action = DebugAction(fixed_code=request.fixed_code, use_hint=request.use_hint)
    result = env.step(action, session_id=session_id)
    result["observation"] = result["observation"].model_dump()
    return result


@app.get("/state/{session_id}")
def state(session_id: str):
    return env.state(session_id=session_id).model_dump()


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
    