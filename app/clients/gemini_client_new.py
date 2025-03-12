import google.generativeai as genai
import asyncio
from typing import AsyncGenerator, Dict, Any, List
from app.utils.logger import logger


class GeminiClientNew:
    """使用官方google-genai库的Gemini API客户端"""

    def __init__(self, api_key: str, proxy: str = None):
        """初始化Gemini客户端

        Args:
            api_key: API密钥
            proxy: 代理服务器地址
        """
        self.api_key = api_key
        self.proxy = proxy
        # 初始化token计数
        self.token_count = 0
        
        # 配置API密钥
        genai.configure(api_key=api_key)
        
        # 如果有代理，设置代理环境变量
        if proxy:
            # 注意：google-genai库使用requests，会自动读取环境变量中的代理设置
            # 在实际使用时，应在应用启动前设置环境变量
            import os
            os.environ['HTTP_PROXY'] = proxy
            os.environ['HTTPS_PROXY'] = proxy

    async def stream_chat(
        self, messages: List[Dict[str, str]], model: str = "gemini-2.0-flash", model_arg: tuple[float, float, float, float] = (0.7, 1.0, 0.0, 0.0)
    ) -> AsyncGenerator[str, None]:
        """流式获取Gemini API的回复

        Args:
            messages: 消息列表
            model: 模型名称，默认为gemini-2.0-flash
            model_arg: 模型参数 (temperature, top_p, presence_penalty, frequency_penalty)

        Yields:
            生成的内容片段
        """
        temperature, top_p, _, _ = model_arg
        
        # 将OpenAI格式的消息转换为Gemini格式
        gemini_messages = self._convert_messages_to_gemini_format(messages)
        
        try:
            # 获取模型
            gemini_model = genai.GenerativeModel(model)
            
            # 设置生成参数
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": 8192  # 设置较大的最大输出长度
                # 注意：google-genai库中stream参数不在generation_config中
                # 而是作为send_message方法的单独参数
            }
            
            # 创建聊天会话
            chat = gemini_model.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
            
            # 获取最后一条消息作为当前提问
            current_message = gemini_messages[-1] if gemini_messages else {"role": "user", "parts": [{"text": ""}]}
            current_content = current_message["parts"][0]["text"] if current_message["parts"] else ""
            
            # 使用异步方式处理流式响应
            response = chat.send_message(current_content, generation_config=generation_config, stream=True)
            
            # 使用asyncio.to_thread将同步API转换为异步
            async def process_stream():
                total_tokens = 0
                for chunk in response:
                    if hasattr(chunk, 'text'):
                        # 估算token数量（简单估算，每个字符约0.33个token）
                        total_tokens += len(chunk.text) // 3 + 1
                        self.token_count = total_tokens
                        yield chunk.text
                
                # 获取完整响应的token统计（如果API支持）
                if hasattr(response, 'usage') and response.usage:
                    if hasattr(response.usage, 'total_tokens'):
                        self.token_count = response.usage.total_tokens
            
            # 返回异步生成器
            async for text in process_stream():
                yield text
                
        except Exception as e:
            logger.error(f"与Gemini API通信时出错: {e}")
            raise
    
    def _convert_messages_to_gemini_format(self, messages: List[Dict[str, str]]) -> List[Dict]:
        """将OpenAI格式的消息转换为Gemini格式

        Args:
            messages: OpenAI格式的消息列表

        Returns:
            Gemini格式的消息列表
        """
        gemini_messages = []
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "user":
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                gemini_messages.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })
            elif role == "system":
                # Gemini不直接支持system角色，将其作为用户消息处理
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": f"System instruction: {content}"}]
                })
        
        return gemini_messages