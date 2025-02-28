# server.py
from fastapi import FastAPI, HTTPException
from loguru import logger
import tomllib
import re
import json
from pathlib import Path
from llm.vllm import VllmClient
from pydantic import BaseModel
from transformers import AutoTokenizer
from tqdm import tqdm


CHECK_OUTPUT_JSON_FORMAT = json.dumps(
    {
        "检查结论": "(直接陈述您的结论，确保简洁明了)",
        "风险等级": "(提供问题相关的风险等级，使用清晰的评级，如“无风险”、“低风险”、“中风险”、“高风险”)",
        "风险依据": "(解释支持您结论的风险依据，包括数据、研究或逻辑推理)",
        "优化建议": "(提供优化建议，包括更改方案、降低风险等等)"
    },
    ensure_ascii=False
)

# 加载配置
with open(Path(__file__).parent / "config.toml", "rb") as f:
    config = tomllib.load(f)
    
HOST = config["server"]["host"]
PORT = config["server"]["port"]

VLLM_HOST_QWEN = config["vllm-qwen25-72b"]["host"]
VLLM_PORT_QWEN = config["vllm-qwen25-72b"]["port"]
MAX_TOKENS_QWEN = config["vllm-qwen25-72b"]["max_tokens"]
TEMPERATURE_QWEN = config["vllm-qwen25-72b"]["temperature"]

VLLM_HOST_DEEPSEEK = config["vllm-deepseek-r1-32b"]["host"]
VLLM_PORT_DEEPSEEK = config["vllm-deepseek-r1-32b"]["port"]
MAX_TOKENS_DEEPSEEK = config["vllm-deepseek-r1-32b"]["max_tokens"]
TEMPERATURE_DEEPSEEK = config["vllm-deepseek-r1-32b"]["temperature"]

CORRECTION_TOKEN_LIMIT = config["app"]["correction"]["token_limit"]
SUMMARY_TOKEN_LIMIT = config["app"]["summary"]["token_limit"]

vllm_qwen = VllmClient(
    VLLM_HOST_QWEN, 
    VLLM_PORT_QWEN, 
    MAX_TOKENS_QWEN,
    TEMPERATURE_QWEN
)

vllm_ds = VllmClient(
    VLLM_HOST_DEEPSEEK, 
    VLLM_PORT_DEEPSEEK, 
    MAX_TOKENS_DEEPSEEK,
    TEMPERATURE_DEEPSEEK
)

# 创建日志目录
LOG_DIR = "./logs"
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# 配置loguru
logger.add(
    f"{LOG_DIR}/service.log",
    rotation="3 MB",       # 每3MB分割新文件
    retention=10,          # 保留最近10个文件
    compression="zip",     # 旧日志压缩保存
    enqueue=True,          # 线程安全
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    backtrace=True,        # 记录异常堆栈
    diagnose=True,         # 显示变量值
    level="INFO"
)

tokenizer_path = Path.cwd() / "model" / "qwen25-72b-tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) 

app = FastAPI()

def parse_json(text):
    # 使用正则表达式提取JSON字符串
    json_match = re.search(r'```json(.*?)```', text, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1)
        try:
            # 解析JSON字符串
            json_obj = json.loads(json_str)
            return json_obj
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return None
    else:
        print("未找到JSON字符串")
        return None


class Request(BaseModel):
    prompt: str
    system_prompt: str
    temperature: float = 0.0


class Response(BaseModel):
    result: str

    
def count_tokens(prompt: str) -> int:
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    return len(tokens)

@app.post("/correction", response_model=Response)
async def correction(request: Request):
    try:
        prompt = request.prompt
        system_prompt = request.system_prompt
        temperature = request.temperature
        token_limit = CORRECTION_TOKEN_LIMIT
        prompt_list = prompt.split("\n")
        tokens_count_total = 0
        prompt_total = ""
        result_total = ""
        
        for prompt in prompt_list:
            tokens_count = count_tokens(prompt)
            
            if tokens_count_total + tokens_count > token_limit:
                logger.info(f"Surpass token limit: {token_limit}")
                result = vllm_qwen.chat(
                    prompt=prompt_total, 
                    system_prompt=system_prompt
                )
                result_total += result + "\n"
                prompt_total = prompt + "\n"
                tokens_count_total = tokens_count
            else:
                prompt_total += prompt + "\n"
                tokens_count_total += tokens_count
        
        result = vllm_qwen.chat(
            prompt=prompt_total, 
            system_prompt=system_prompt,
            temperature=temperature
        )
        result_total += result

        return Response(result=result_total.strip())
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summary", response_model=Response)
async def summary(request: Request):
    try:
        prompt = request.prompt
        system_prompt = request.system_prompt
        temperature = request.temperature
        token_limit = SUMMARY_TOKEN_LIMIT
        prompt_list = prompt.split("\n")
        tokens_count_total = 0
        prompt_accumulate = ""
        result = ""
        
        for prompt in prompt_list:
            tokens_count = count_tokens(prompt)
            
            if tokens_count_total + tokens_count > token_limit:
                logger.info(f"Surpass token limit: {token_limit}")
                prompt_total = f"[现有摘要]: \n{result}\n\n[额外通话内容]: \n{prompt_accumulate}"
                result = vllm_qwen.chat(
                    prompt=prompt_total, 
                    system_prompt=system_prompt,
                    temperature=temperature
                )
                prompt_accumulate = prompt + "\n"
                tokens_count_total = tokens_count
            else:
                prompt_accumulate += prompt + "\n"
                tokens_count_total += tokens_count
        
        prompt_total = f"[现有摘要]: \n{result}\n\n[额外内容]: \n{prompt_accumulate}"
        result = vllm_qwen.chat(
            prompt=prompt_total, 
            system_prompt=system_prompt,
            temperature=temperature
        )
        return Response(result=result.strip())
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


class Scenario(BaseModel):
    scenario_name: str
    scenario_description: str | None = None
    
class KeyPoint(BaseModel):
    key_point_name: str
    key_point_description: str | None = None
    scenarios: list[Scenario] | None = None

class Task(BaseModel):
    task_name: str
    task_description: str | None = None
    key_points: list[KeyPoint]

class CheckTaskRequest(BaseModel):
    task: Task
    content: str
    content_type: str
    system_prompt_scenario: str
    system_prompt_task: str
    
class CheckTaskResponse(BaseModel):
    检查结论: str
    风险等级: str
    风险依据: str
    优化建议: str
    场景检查结果: str

@app.post("/check-task", response_model=CheckTaskResponse)
async def check_task(request: CheckTaskRequest):
    try:        
        system_prompt_list = []
        for i in range(len(request.task.key_points)):
            for j in range(len(request.task.key_points[i].scenarios)):
                system_prompt_scenario = request.system_prompt_scenario.format(
                    content_type = request.content_type,
                    key_point_name = request.task.key_points[i].key_point_name,
                    key_point_description = request.task.key_points[i].key_point_description,
                    scenario_name = request.task.key_points[i].scenarios[j].scenario_name,
                    scenario_description = request.task.key_points[i].scenarios[j].scenario_description
                )
                system_prompt_list.append(system_prompt_scenario)
                
        check_result_scenarios = ""
        for system_prompt in tqdm(system_prompt_list):
            result = vllm_ds.chat(
                prompt=request.content, 
                system_prompt=system_prompt,
                temperature=0.0
            )
            
            # 如果是R1模型，则去掉思考的部分
            if "</think>" in result:
                result = result.split("</think>")[-1]
                
            check_result_scenarios += f"{result}\n---\n"
            
        check_result_scenarios.strip("\n---\n").strip()
            
        system_prompt_task = request.system_prompt_task.format(
            content_type = request.content_type,
            task_name = request.task.task_name,
            task_description = request.task.task_description,
            check_result_scenarios = check_result_scenarios,
            output_json_format = "```json\n" + CHECK_OUTPUT_JSON_FORMAT + "\n```"
        )
        
        print(system_prompt_task)
        
        result = vllm_qwen.chat(
            prompt=request.content, 
            system_prompt=system_prompt_task,
            max_tokens= 2_000,
            temperature=0.0
        )
        
        try:
            result_json = parse_json(result)
        except Exception as e:
            logger.info("Failed to parse JSON")
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))
        
        result_json['场景检查结果'] = check_result_scenarios
        
        return CheckTaskResponse.model_validate(
            result_json
        )
        
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app", 
        host=config["server"]["host"], 
        port=config["server"]["port"], 
        reload=True,
        timeout_keep_alive=1800
    )