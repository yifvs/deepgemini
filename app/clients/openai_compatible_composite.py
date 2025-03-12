import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Any, List

from app.clients.deepseek_client import DeepSeekClient
from app.clients.openai_compatible_client import OpenAICompatibleClient
from app.clients.gemini_client_new import GeminiClientNew
from app.utils.logger import logger


class OpenAICompatibleComposite:
    """处理 DeepSeek 和其他 OpenAI 兼容模型的流式输出衔接"""

    def __init__(
        self,
        deepseek_api_key: str,
        openai_api_key: str,
        deepseek_api_url: str = "https://api.deepseek.com/v1/chat/completions",
        openai_api_url: str = "",  # 将由具体实现提供
        is_origin_reasoning: bool = True,
        proxy: str = None,
    ):
        # 初始化token计数器
        self.deepseek_tokens = 0
        self.gemini_tokens = 0
        """初始化 API 客户端

        Args:
            deepseek_api_key: DeepSeek API密钥
            openai_api_key: OpenAI 兼容服务的 API密钥
            deepseek_api_url: DeepSeek API地址
            openai_api_url: OpenAI 兼容服务的 API地址
            is_origin_reasoning: 是否使用原始推理过程
            proxy: 代理服务器地址
        """
        self.deepseek_client = DeepSeekClient(deepseek_api_key, deepseek_api_url, proxy=proxy)
        
        # 根据目标模型选择合适的客户端
        if "gemini" in openai_api_url:
            # 使用新的基于google-genai库的客户端
            self.openai_client = GeminiClientNew(openai_api_key, proxy=proxy)
        else:
            self.openai_client = OpenAICompatibleClient(openai_api_key, openai_api_url, proxy=proxy)
            
        self.is_origin_reasoning = is_origin_reasoning

    async def chat_completions_with_stream(
        self,
        messages: List[Dict[str, str]],
        model_arg: tuple[float, float, float, float],
        deepseek_model: str = "deepseek-reasoner",
        target_model: str = "",
    ) -> AsyncGenerator[bytes, None]:
        """处理完整的流式输出过程

        Args:
            messages: 初始消息列表
            model_arg: 模型参数 (temperature, top_p, presence_penalty, frequency_penalty)
            deepseek_model: DeepSeek 模型名称
            target_model: 目标 OpenAI 兼容模型名称

        Yields:
            字节流数据，格式如下：
            {
                "id": "chatcmpl-xxx",
                "object": "chat.completion.chunk",
                "created": timestamp,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": reasoning_content,
                        "content": content
                    }
                }]
            }
        """
        # 生成唯一的会话ID和时间戳
        chat_id = f"chatcmpl-{hex(int(time.time() * 1000))[2:]}"
        created_time = int(time.time())

        # 创建队列，用于收集输出数据
        output_queue = asyncio.Queue()
        # 队列，用于传递 DeepSeek 推理内容
        reasoning_queue = asyncio.Queue()

        # 用于存储 DeepSeek 的推理累积内容
        reasoning_content = []

        async def process_deepseek():
            logger.info(f"开始处理 DeepSeek 流，使用模型：{deepseek_model}")
            try:
                async for content_type, content in self.deepseek_client.stream_chat(
                    messages, deepseek_model, self.is_origin_reasoning
                ):
                    # 获取DeepSeek的token使用情况
                    if hasattr(self.deepseek_client, 'token_count'):
                        self.deepseek_tokens = self.deepseek_client.token_count
                    if content_type == "reasoning":
                        reasoning_content.append(content)
                        response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": deepseek_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "reasoning_content": content,
                                        "content": "",
                                    },
                                }
                            ],
                        }
                        await output_queue.put(
                            f"data: {json.dumps(response)}\n\n".encode("utf-8")
                        )
                    elif content_type == "content":
                        # 当收到 content 类型时，将完整的推理内容发送到 reasoning_queue
                        logger.info(
                            f"DeepSeek 推理完成，收集到的推理内容长度：{len(''.join(reasoning_content))}"
                        )
                        await reasoning_queue.put("".join(reasoning_content))
                        break
            except Exception as e:
                logger.error(f"处理 DeepSeek 流时发生错误: {e}")
                await reasoning_queue.put("")
            # 标记 DeepSeek 任务结束
            logger.info("DeepSeek 任务处理完成，标记结束")
            await output_queue.put(None)

        async def process_openai():
            try:
                logger.info("等待获取 DeepSeek 的推理内容...")
                reasoning = await reasoning_queue.get()
                logger.debug(
                    f"获取到推理内容，内容长度：{len(reasoning) if reasoning else 0}"
                )
                if not reasoning:
                    logger.warning("未能获取到有效的推理内容，将使用默认提示继续")
                    reasoning = "获取推理内容失败"

                # 构造 OpenAI 的输入消息
                openai_messages = messages.copy()
                combined_content = f"""
                Here's my another model's reasoning process:\n{reasoning}\n\n
                Based on this reasoning, please provide a comprehensive and detailed response. Your answer should be thorough and complete, covering all aspects of the question. Don't be too brief - aim for a substantial explanation that fully addresses the query:"""

                # 检查过滤后的消息列表是否为空
                if not openai_messages:
                    raise ValueError("消息列表为空，无法处理请求")

                # 获取最后一个消息并检查其角色
                last_message = openai_messages[-1]
                if last_message.get("role", "") != "user":
                    raise ValueError("最后一个消息的角色不是用户，无法处理请求")

                # 修改最后一个消息的内容
                original_content = last_message["content"]
                fixed_content = f"Here's my original input:\n{original_content}\n\n{combined_content}"
                last_message["content"] = fixed_content

                logger.info(f"开始处理 OpenAI 兼容模型流，使用模型：{target_model}")
                async for content in self.openai_client.stream_chat(
                    openai_messages, target_model, model_arg
                ):
                    # 获取Gemini的token使用情况
                    if isinstance(self.openai_client, GeminiClientNew) and hasattr(self.openai_client, 'token_count'):
                        self.gemini_tokens = self.openai_client.token_count
                        
                    response = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": target_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "reasoning_content": "",
                                    "content": content,
                                },
                            }
                        ],
                    }
                    await output_queue.put(
                        f"data: {json.dumps(response)}\n\n".encode("utf-8")
                    )
            except Exception as e:
                logger.error(f"处理 OpenAI 兼容模型流时发生错误: {e}")
            finally:
                # 标记 OpenAI 任务结束
                logger.info("OpenAI 兼容模型任务处理完成，标记结束")
                await output_queue.put(None)

        # 创建并启动两个任务
        deepseek_task = asyncio.create_task(process_deepseek())
        openai_task = asyncio.create_task(process_openai())

        # 等待任务完成并输出结果
        tasks_done = 0
        while tasks_done < 2:  # 两个任务都需要完成
            item = await output_queue.get()
            if item is None:
                tasks_done += 1
                continue
            yield item

        # 发送结束标记
        yield f"data: [DONE]\n\n".encode("utf-8")

        # 确保任务已完成
        await asyncio.gather(deepseek_task, openai_task, return_exceptions=True)