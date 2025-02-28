import requests
import json
import time
import tomllib
from pathlib import Path


# 读取配置
with open(Path(__file__).parent.parent / "config.toml", "rb") as f:
    config = tomllib.load(f)


VLLM_HOST_QWEN = config["vllm-qwen25-72b"]["host"]
VLLM_PORT_QWEN = config["vllm-qwen25-72b"]["port"]
MAX_TOKENS_QWEN = config["vllm-qwen25-72b"]["max_tokens"]
TEMPERATURE_QWEN = config["vllm-qwen25-72b"]["temperature"]

VLLM_HOST_DEEPSEEK = config["vllm-deepseek-r1-32b"]["host"]
VLLM_PORT_DEEPSEEK = config["vllm-deepseek-r1-32b"]["port"]
MAX_TOKENS_DEEPSEEK = config["vllm-deepseek-r1-32b"]["max_tokens"]
TEMPERATURE_DEEPSEEK = config["vllm-deepseek-r1-32b"]["temperature"]


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行被装饰的函数
        end_time = time.time()  # 记录结束时间
        execution_time = end_time - start_time  # 计算执行时间
        print(f"函数 {func.__name__} 执行时间: {execution_time:.4f} 秒")
        return result
    return wrapper


class VllmClient:
    def __init__(
        self, 
        host, 
        port,
        max_tokens,
        temperature
    ):
        self.host = host
        self.port = port
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def chat(
        self,
        prompt: str, 
        system_prompt: str = "", 
        history: list | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None
    ):
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
        
        url = f"http://{self.host}:{self.port}/chat"  # Adjust the URL if needed
        
        payload = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "history": history or [],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()["content"]

    def chat_stream(
        self,
        prompt, 
        system_prompt="", 
        history=None,
        max_tokens: int | None = None,
        temperature: float | None = None
    ):
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
        
        url = f"http://{self.host}:{self.port}/chat-stream"  # Adjust the URL if needed
        
        payload = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "history": history or [],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                yield chunk["content"]
            
# Example usage
if __name__ == "__main__":
    vllm = VllmClient(
        VLLM_HOST_QWEN, 
        VLLM_PORT_QWEN, 
        MAX_TOKENS_QWEN,
        TEMPERATURE_QWEN
    )
    
    # vllm = VllmClient(
    #     VLLM_HOST_DEEPSEEK, 
    #     VLLM_PORT_DEEPSEEK, 
    #     MAX_TOKENS_DEEPSEEK,
    #     TEMPERATURE_DEEPSEEK
    # )
    
    system_prompt = "你是一个AI助手"
    user_prompt = "写一篇100字的高中作文，关于信念。"
    
    # print("Testing streaming chat:")
    # full_response = ""

    # start = time.time()

    # for content in vllm.chat_stream(user_prompt, system_prompt):
    #     print(content, end='', flush=True)
    #     full_response += content
        
    # duration = time.time() - start
    # print(f"Duration: {duration:.2f}")

    print("\n\nTesting non-streaming chat:")
    non_stream_response = vllm.chat(user_prompt, system_prompt)
    print("Non-streaming response:", non_stream_response)
