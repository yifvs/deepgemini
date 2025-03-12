import json
import aiohttp
from typing import AsyncGenerator, Dict, Tuple, Any, List
from app.utils.logger import logger


class DeepSeekClient:
    """DeepSeek API客户端，处理与DeepSeek API的交互"""

    def __init__(self, api_key: str, api_url: str, proxy: str = None):
        """初始化DeepSeek客户端

        Args:
            api_key: DeepSeek API密钥
            api_url: DeepSeek API地址
            proxy: 代理服务器地址
        """
        self.api_key = api_key
        # 验证并修正API URL格式
        api_url = api_url.strip()
        # 确保URL以http或https开头
        if not (api_url.startswith('http://') or api_url.startswith('https://')):
            api_url = 'https://' + api_url
        # 确保URL以/v1/chat/completions结尾
        if not api_url.endswith('/v1/chat/completions'):
            # 移除可能的尾部斜杠
            api_url = api_url.rstrip('/')
            # 添加正确的路径
            api_url = f"{api_url}/v1/chat/completions"
        
        self.api_url = api_url
        self.proxy = proxy
        self.token_count = 0
        logger.info(f"初始化DeepSeek客户端，API地址: {self.api_url}")

    async def stream_chat(
        self, messages: List[Dict[str, str]], model: str, is_origin_reasoning: bool = True
    ) -> AsyncGenerator[Tuple[str, str], None]:
        # 初始化token计数
        token_count = 0
        """流式获取DeepSeek的回复

        Args:
            messages: 消息列表
            model: 模型名称
            is_origin_reasoning: 是否使用原始推理过程

        Yields:
            (content_type, content) 元组，content_type可能是"reasoning"或"content"
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 如果使用原始推理过程，修改最后一条用户消息
        if is_origin_reasoning and messages and messages[-1]["role"] == "user":
            original_content = messages[-1]["content"]
            messages[-1]["content"] = f"请用思维链（Chain of Thought）的方式分析以下问题，一步一步地思考：\n\n问题：{original_content}\n\n思考过程："

        data = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 8192,
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
                        logger.error(f"DeepSeek API返回错误: {response.status}, {error_text}")
                        raise Exception(f"DeepSeek API返回错误: {response.status}, {error_text}")

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
                                
                                # 获取token使用情况
                                usage = data.get("usage", {})
                                if usage:
                                    token_count = usage.get("total_tokens", 0)
                                    self.token_count = token_count
                                
                                if content:
                                    # 在原始推理模式下，所有内容都视为推理过程
                                    if is_origin_reasoning:
                                        yield "reasoning", content
                                    else:
                                        yield "content", content
                            except json.JSONDecodeError as e:
                                logger.error(f"解析DeepSeek响应时出错: {e}, line: {line}")

            # 推理完成后，如果是原始推理模式，发送一个空的content信号
            if is_origin_reasoning:
                yield "content", ""

        except Exception as e:
            logger.error(f"与DeepSeek API通信时出错: {e}")
            raise