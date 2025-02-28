# client.py
import requests
import tomllib
from pathlib import Path
from pydantic import BaseModel
import time
import json


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

def load_check_scenario_prompts() -> tuple[list[str], str]:
    system_prompt_path = Path(__file__).parent / "prompts" / "check" / "system_prompt_scenario.txt"
    content_path = Path(__file__).parent / "prompts" / "check" / "content.txt"
    
    with open(system_prompt_path, "r") as f:
        system_prompt_template = f.read()
    with open(content_path, "r") as f:
        content = f.read()
        
    with open(Path(__file__).parent / "prompts" / "check" / "task.json", "r") as f:
        task = json.load(f)['task']
        
    system_prompt_list = []
        
    for i in range(len(task['key_points'])):
        for j in range(len(task['key_points'][i]['scenarios'])):
            system_prompt = system_prompt_template.format(
                key_point_name = task['key_points'][i]['key_point_name'],
                key_point_description = task['key_points'][i]['key_point_description'],
                scenario_name=task['key_points'][i]['scenarios'][j]['scenario_name'],
                scenario_description=task['key_points'][i]['scenarios'][j]['scenario_description']
            )
            system_prompt_list.append(system_prompt)
    
    return system_prompt_list, content

@timer
def test_correction():
    media_type = "call"
    # media_type = "meeting"
    # media_type = "stream_and_video"
    # media_type = "social_media"
    
    system_prompt, prompt = load_correction_prompts(media_type)
    url = f"{BASE_URL}/correction"
    
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
    
    # 如果是R1模型，则去掉思考的部分
    # if "</think>" in result:
    #     result = result.split("</think>")[-1]
    
    print(result)
    
@timer
def test_summary():
    # media_type = "call"
    media_type = "meeting"
    
    system_prompt, prompt = load_summary_prompts(media_type)
    url = f"{BASE_URL}/summary"
    
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
    
    # 如果是R1模型，则去掉思考的部分
    # if "</think>" in result:
    #     result = result.split("</think>")[-1]
    
    print(result)

@timer
def test_check():
    url = f"{BASE_URL}/check-task"
    
    with open(Path(__file__).parent / "prompts" / "check" / "system_prompt_task.txt", "r") as f:
        system_prompt_task = f.read()
    with open(Path(__file__).parent / "prompts" / "check" / "system_prompt_scenario.txt", "r") as f:
        system_prompt_scenario = f.read()
    with open(Path(__file__).parent / "prompts" / "check" / "content.txt", "r") as f:
        content = f.read()
    with open(Path(__file__).parent / "prompts" / "check" / "check_task_request.json", "r") as f:
        request_data = json.load(f)
    
    request_data['system_prompt_task'] = system_prompt_task
    request_data['system_prompt_scenario'] = system_prompt_scenario
    request_data['content'] = content
    
    response = requests.post(
        url=url,
        json=request_data,
        headers={"Content-Type": "application/json"}
    )
    response.raise_for_status()
    
    with open(f"output/check_task_output_{time.strftime('%H_%M_%S')}.json", "w") as f:
        json.dump(response.json(), f, indent=4, ensure_ascii=False)

    
if __name__ == "__main__":
    BASE_URL = "http://localhost:8000"
    # BASE_URL = "http://10.101.100.13:8010"
    
    # test_correction()
    # test_summary()
    test_check()