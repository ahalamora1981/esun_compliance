# client.py
import requests
import tomllib
from pathlib import Path
from pydantic import BaseModel
import time


with open(Path(__file__).parent / "config.toml", "rb") as f:
    config = tomllib.load(f)


class Request(BaseModel):
    prompt: str
    system_prompt: str
    temperature: float = 0.0


class Response(BaseModel):
    result: str

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Execution time: {end_time - start_time:0.4f} seconds")
        return result
    return wrapper

def load_correction_prompts(media_type: str) -> tuple[str, str]:
    system_prompt_path = Path(__file__).parent / "prompts" / "correction" / media_type / "system_prompt.txt"
    user_prompt_path = Path(__file__).parent / "prompts" / "correction" / media_type / "user_prompt.txt"
    
    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()
    with open(user_prompt_path, "r") as f:
        prompt = f.read()
    
    return system_prompt, prompt

def load_summary_prompts(media_type: str) -> tuple[str, str]:
    system_prompt_path = Path(__file__).parent / "prompts" / "summary" / media_type / "system_prompt.txt"
    user_prompt_path = Path(__file__).parent / "prompts" / "summary" / media_type / "user_prompt.txt"
    
    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()
    with open(user_prompt_path, "r") as f:
        prompt = f.read()
    
    return system_prompt, prompt

def load_check_scenario_prompts(scenario_name: str, scenario_description: str) -> tuple[str, str]:
    system_prompt_path = Path(__file__).parent / "prompts" / "check" / "scenario" / "system_prompt.txt"
    user_prompt_path = Path(__file__).parent / "prompts" / "check" / "scenario" / "user_prompt.txt"
    
    with open(system_prompt_path, "r") as f:
        system_prompt = f.read()
    with open(user_prompt_path, "r") as f:
        prompt = f.read()
        
    system_prompt = system_prompt.format(
        scenarioName=scenario_name,
        scenarioDescription=scenario_description
    )
    
    return system_prompt, prompt

@timer
def main():
    # BASE_URL = "http://localhost:8000"
    BASE_URL = "http://10.101.100.13:8010"
    
    ### Correction ### 
    # media_type = "call"
    media_type = "meeting"
    # media_type = "stream_and_video"
    # media_type = "social_media"
    
    system_prompt, prompt = load_correction_prompts(media_type)
    url = f"{BASE_URL}/correction"
    ### End Correction ### 

    ### Summary ###
    # # media_type = "call"
    # media_type = "meeting"
    
    # system_prompt, prompt = load_summary_prompts(media_type)
    # url = f"{BASE_URL}/summary"
    ### End Summary ###
    
    ### Check Scenario ###
    # scenario_name = "虚假宣传检查"
    # scenario_description = "- 根据背景信息，检查内容中是否有虚假宣传"
    
    # system_prompt, prompt = load_check_scenario_prompts(scenario_name, scenario_description)
    # url = f"{BASE_URL}/check-scenario"
    ### End Check Scenario ###
    
    request = Request(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=0.0
    )

    response = requests.post(
        url=url,
        json=request.model_dump(),
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()

    response_model = Response.model_validate(response.json())
    result = response_model.result
    
    if "</think>" in result:
        result = result.split("</think>")[-1]
    
    print(result)
    
if __name__ == "__main__":
    main()