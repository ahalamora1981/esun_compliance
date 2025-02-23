# client.py
import requests
import tomllib
from pathlib import Path
from pydantic import BaseModel


with open(Path(__file__).parent / "config.toml", "rb") as f:
    config = tomllib.load(f)


class CorrectionRequest(BaseModel):
    prompt: str
    system_prompt: str


class CorrectionResponse(BaseModel):
    result: str


if __name__ == "__main__":
    application = "correction"
    
    # media_type = "call"
    media_type = "meeting"
    
    system_prompt_path = Path(__file__).parent / "prompts" / application / media_type / "system_prompt.txt"
    user_prompt_path = Path(__file__).parent / "prompts" / application / media_type / "user_prompt.txt"
    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()
    with open(user_prompt_path, "r") as f:
        prompt = f.read()

    correction_request = CorrectionRequest(
        prompt=prompt,
        system_prompt=system_prompt
    )

    response = requests.post(
        url=f"http://localhost:8000/{application}",
        json=correction_request.model_dump(),
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()

    response_model = CorrectionResponse.model_validate(response.json())
    print(response_model.result)