import streamlit as st
import os
import asyncio
import requests
import json
import time
import aiohttp
from dotenv import load_dotenv
from app.clients.openai_compatible_composite import OpenAICompatibleComposite
from app.utils.logger import logger

# 获取代理设置（仍从环境变量获取）
load_dotenv()
HTTP_PROXY = os.getenv("HTTP_PROXY")

# 初始化模型
def init_composite_model(deepseek_api_key, gemini_api_key, deepseek_api_url):
    # 初始化组合模型服务
    composite = OpenAICompatibleComposite(
        deepseek_api_key=deepseek_api_key,
        openai_api_key=gemini_api_key,
        deepseek_api_url=deepseek_api_url,
        openai_api_url="gemini",
        is_origin_reasoning=True,
        proxy=HTTP_PROXY
    )
    return composite

# 运行异步函数的辅助函数
def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# 设置页面
st.set_page_config(page_title="DeepSeek + Gemini 缝合怪", page_icon="🤖", layout="wide")

# 添加CSS样式使标题和介绍固定在顶部
st.markdown("""
<style>
    .fixed-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background-color: white;
        z-index: 999;
        padding: 10px 5% 10px 5%;
        border-bottom: 1px solid #f0f0f0;
        width: 100%;
    }
    .main-content {
        margin-top: 150px; /* 为固定头部留出空间 */
    }
</style>
""", unsafe_allow_html=True)

# 创建固定在顶部的标题和介绍
with st.container():
    st.markdown('<div class="fixed-header">', unsafe_allow_html=True)
    st.title("DeepSeek + Gemini 缝合怪")
    st.markdown("""本项目先使用DeepSeek的思维链分析问题，然后再用Gemini总结输出结果。""")
    st.markdown('</div>', unsafe_allow_html=True)

# 创建主内容区域，添加适当的上边距
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 初始化API密钥会话状态
if "deepseek_api_key" not in st.session_state:
    st.session_state.deepseek_api_key = ""
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""
if "deepseek_api_url" not in st.session_state:
    st.session_state.deepseek_api_url = "https://api.deepseek.com/v1/chat/completions"

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
prompt = st.chat_input("请输入您的问题...")

# 处理用户输入
if prompt:
    # 显示用户消息
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.spinner("DeepSeek正在思考中..."):
        try:
            # 获取API密钥
            deepseek_api_key = st.session_state.deepseek_api_key
            gemini_api_key = st.session_state.gemini_api_key
            deepseek_api_url = st.session_state.deepseek_api_url
            
            # 检查API密钥
            if not deepseek_api_key or not gemini_api_key:
                if not deepseek_api_key:
                    st.error("请在侧边栏设置DeepSeek API密钥")
                if not gemini_api_key:
                    st.error("请在侧边栏设置Gemini API密钥")
            else:
                # 初始化组合模型
                composite_model = init_composite_model(deepseek_api_key, gemini_api_key, deepseek_api_url)
                
                # 准备消息
                messages = [{"role": "user", "content": prompt}]
                
                # 模型参数 (temperature, top_p, presence_penalty, frequency_penalty)
                model_arg = (0.7, 1.0, 0.0, 0.0)
                
                # 存储思维过程和最终答案
                reasoning_content = []
                final_answer = []
                
                # 创建两个容器用于显示内容
                reasoning_container = st.empty()
                answer_container = st.empty()
                
                # 定义异步处理函数
                async def process_stream():
                    # 记录开始时间
                    start_time = time.time()
                    
                    async for chunk in composite_model.chat_completions_with_stream(
                        messages=messages,
                        model_arg=model_arg,
                        deepseek_model="deepseek-ai/DeepSeek-R1" if "siliconflow.cn" in deepseek_api_url else "deepseek-reasoner",
                        target_model="gemini-2.0-flash"
                    ):
                        # 解析响应
                        if chunk.startswith(b"data: ") and not chunk.startswith(b"data: [DONE]"):
                            try:
                                data = json.loads(chunk[6:].decode("utf-8"))
                                delta = data.get("choices", [{}])[0].get("delta", {})
                                
                                # 处理推理内容
                                reasoning = delta.get("reasoning_content", "")
                                if reasoning:
                                    reasoning_content.append(reasoning)
                                    with reasoning_container.container():
                                        with st.chat_message("assistant"):
                                            with st.expander("查看思维过程", expanded=False):
                                                st.markdown("".join(reasoning_content))
                                
                                # 处理最终答案
                                content = delta.get("content", "")
                                if content:
                                    final_answer.append(content)
                                    # 计算耗时
                                    elapsed_time = time.time() - start_time
                                    # 获取token使用情况
                                    deepseek_tokens = composite_model.deepseek_tokens
                                    gemini_tokens = composite_model.gemini_tokens
                                    # 添加统计信息
                                    stats = f"\n\n---\n**统计信息**\n- 耗时：{elapsed_time:.2f}秒\n- DeepSeek消耗tokens：{deepseek_tokens}\n- Gemini消耗tokens：{gemini_tokens}"
                                    with answer_container.container():
                                        with st.chat_message("assistant"):
                                            st.markdown("".join(final_answer) + stats)
                            except json.JSONDecodeError as e:
                                logger.error(f"解析响应时出错: {e}")
                
                # 运行异步处理
                run_async(process_stream())
                
                # 将最终答案添加到会话状态
                if final_answer:
                    st.session_state.messages.append({"role": "assistant", "content": "".join(final_answer)})
        except Exception as e:
            st.error(f"发生错误: {str(e)}")
            logger.error(f"处理请求时出错: {e}")

# 关闭主内容区域的div
st.markdown('</div>', unsafe_allow_html=True)

# 侧边栏 - API密钥设置
with st.sidebar:
    st.title("API设置")
    
    # DeepSeek API设置
    st.subheader("DeepSeek API设置")
    deepseek_api_key = st.text_input("DeepSeek API密钥", value=st.session_state.deepseek_api_key, type="password")
    deepseek_api_url = st.text_input("DeepSeek API地址", value=st.session_state.deepseek_api_url)
        
    # 测试DeepSeek API连接按钮
    if st.button("测试DeepSeek连接"):
        if not deepseek_api_key:
            st.error("请先输入DeepSeek API密钥")
        else:
            with st.spinner("正在测试连接..."):
                try:
                    # 创建一个简单的测试请求
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {deepseek_api_key}"
                    }
                    # 使用与主应用相同的模型名称保持一致
                    # 判断是否使用siliconflow.cn的API
                    if "siliconflow.cn" in deepseek_api_url:
                        model_name = "deepseek-ai/DeepSeek-R1"
                    else:
                        model_name = "deepseek-reasoner"
                    data = {
                        "model": model_name,
                        "messages": [{"role": "user", "content": "测试连接"}],
                        "max_tokens": 5
                    }
                    
                    # 记录使用的模型名称
                    logger.info(f"测试连接使用模型: {model_name}")
                    
                    # 使用异步函数测试连接
                    async def test_connection():
                        try:
                            # 验证并修正API URL格式
                            api_url = deepseek_api_url.strip()
                            # 确保URL以http或https开头
                            if not (api_url.startswith('http://') or api_url.startswith('https://')):
                                api_url = 'https://' + api_url
                            # 确保URL以/v1/chat/completions结尾
                            if not api_url.endswith('/v1/chat/completions'):
                                # 移除可能的尾部斜杠
                                api_url = api_url.rstrip('/')
                                # 添加正确的路径
                                api_url = f"{api_url}/v1/chat/completions"
                            
                            logger.info(f"测试连接到DeepSeek API: {api_url}")
                            
                            async with aiohttp.ClientSession() as session:
                                async with session.post(
                                    api_url,
                                    headers=headers,
                                    json=data,
                                    proxy=HTTP_PROXY,
                                    timeout=10
                                ) as response:
                                    if response.status == 200:
                                        return True, "连接成功！"
                                    else:
                                        error_text = await response.text()
                                        logger.error(f"DeepSeek API返回错误: {response.status}, {error_text}")
                                        return False, f"连接失败: {response.status}, {error_text}"
                        except Exception as e:
                            return False, f"连接错误: {str(e)}"
                    
                    # 运行测试连接的异步函数
                    success, message = run_async(test_connection())
                    
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                except Exception as e:
                    st.error(f"测试连接时发生错误: {str(e)}")
                    logger.error(f"测试连接时发生错误: {e}")
    
    # Gemini API设置
    st.subheader("Gemini API设置")
    gemini_api_key = st.text_input("Gemini API密钥", value=st.session_state.gemini_api_key, type="password")
    
    # 保存按钮
    if st.button("保存设置"):
        st.session_state.deepseek_api_key = deepseek_api_key
        st.session_state.gemini_api_key = gemini_api_key
        st.session_state.deepseek_api_url = deepseek_api_url
        st.success("设置已保存！")
    
    
    # 关于模型的说明
    st.subheader("关于模型")
    st.markdown("""
    - **DeepSeek R1**: 用于生成详细的思维链分析
    - **Gemini 2.0 Flash**: 用于总结和优化最终输出
    """)