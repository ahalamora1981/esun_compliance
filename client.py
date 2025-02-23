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
    with open(Path(__file__).parent / "prompts" / "system_prompt.txt", "r") as f:
        system_prompt = f.read()
    with open(Path(__file__).parent / "prompts" / "user_prompt.txt", "r") as f:
        prompt = f.read()

    correction_request = CorrectionRequest(
        prompt=prompt,
        system_prompt=system_prompt
    )

    response = requests.post(
        url="http://localhost:8000/correction",
        json=correction_request.model_dump(),
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()

    response_model = CorrectionResponse.model_validate(response.json())
    print(response_model.result)