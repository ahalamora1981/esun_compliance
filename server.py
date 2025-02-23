# server.py
from fastapi import FastAPI, HTTPException
from loguru import logger
import tomllib
from pathlib import Path
from llm.vllm import vllm_chat
from pydantic import BaseModel
from transformers import AutoTokenizer


# 加载配置
with open(Path(__file__).parent / "config.toml", "rb") as f:
    config = tomllib.load(f)

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

class CorrectionRequest(BaseModel):
    prompt: str
    system_prompt: str

class CorrectionResponse(BaseModel):
    result: str
    
def count_tokens(prompt: str) -> int:
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    return len(tokens)

@app.post("/correction", response_model=CorrectionResponse)
def correction(
    request: CorrectionRequest
):
    try:
        prompt = request.prompt * 2
        system_prompt = request.system_prompt
        token_limit = config['app']['correction']['token_limit']
        prompt_list = prompt.split("\n")
        tokens_count_total = 0
        prompt_total = ""
        result_total = ""
        
        for prompt in prompt_list:
            tokens_count = count_tokens(prompt)
            
            if tokens_count_total + tokens_count > token_limit:
                logger.info(f"Surpass token limit: {token_limit}")
                result = vllm_chat(
                    prompt=prompt_total, 
                    system_prompt=system_prompt
                )
                result_total += result + "\n"
                prompt_total = prompt + "\n"
                tokens_count_total = tokens_count
            else:
                prompt_total += prompt + "\n"
                tokens_count_total += tokens_count
        
        result = vllm_chat(
            prompt=prompt_total, 
            system_prompt=system_prompt
        )
        result_total += result

        return CorrectionResponse(result=result_total.strip())
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