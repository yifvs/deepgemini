import json
import aiohttp
from typing import AsyncGenerator, Dict, Any, List
from app.utils.logger import logger


class OpenAICompatibleClient:
    """OpenAI兼容API客户端，处理与OpenAI兼容API的交互"""

    def __init__(self, api_key: str, api_url: str, proxy: str = None):
        """初始化OpenAI兼容客户端

        Args:
            api_key: API密钥
            api_url: API地址
            proxy: 代理服务器地址
        """
        self.api_key = api_key
        self.api_url = api_url
        self.proxy = proxy

    async def stream_chat(
        self, messages: List[Dict[str, str]], model: str, model_arg: tuple[float, float, float, float] = (0.7, 1.0, 0.0, 0.0)
    ) -> AsyncGenerator[str, None]:
        """流式获取OpenAI兼容API的回复

        Args:
            messages: 消息列表
            model: 模型名称
            model_arg: 模型参数 (temperature, top_p, presence_penalty, frequency_penalty)

        Yields:
            生成的内容片段
        """
        temperature, top_p, presence_penalty, frequency_penalty = model_arg
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stream": True
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    proxy=self.proxy
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenAI兼容API返回错误: {response.status}, {error_text}")
                        raise Exception(f"OpenAI兼容API返回错误: {response.status}, {error_text}")

                    # 处理流式响应
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if not line or line == "data: [DONE]":
                            continue

                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError as e:
                                logger.error(f"解析OpenAI兼容API响应时出错: {e}, line: {line}")

        except Exception as e:
            logger.error(f"与OpenAI兼容API通信时出错: {e}")
            raise