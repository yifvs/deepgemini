# DeepSeek + Gemini 缝合怪

这是一个基于Streamlit的AI对话应用，它结合了DeepSeek和Gemini两个大语言模型的优势：

- 首先使用**DeepSeek**生成详细的思维链分析
- 然后用**Gemini**对分析结果进行总结和优化输出
  ![image](https://github.com/user-attachments/assets/c97080dc-b2e7-4988-9d19-b2ed13e8214e)


## 📖 项目架构

### 整体架构

本项目采用模块化设计，主要由以下几个部分组成：

1. **前端界面**：基于Streamlit构建的用户交互界面
2. **模型组合器**：`OpenAICompatibleComposite`类，负责协调DeepSeek和Gemini模型的工作流程
3. **模型客户端**：
   - `DeepSeekClient`：处理与DeepSeek API的通信
   - `GeminiClientNew`：基于google-genai库的Gemini API客户端
   - `OpenAICompatibleClient`：通用的OpenAI兼容API客户端
4. **工具类**：日志记录等辅助功能

### 数据流程

1. 用户在Streamlit界面输入问题
2. 问题被发送到`OpenAICompatibleComposite`组合器
3. 组合器首先调用`DeepSeekClient`，向DeepSeek API发送请求
   - 在请求中添加思维链提示，要求模型一步步思考
   - 流式接收并显示DeepSeek的思维过程
4. DeepSeek完成思考后，组合器将完整的思维链结果传递给`GeminiClientNew`
5. `GeminiClientNew`向Gemini API发送请求，包含原始问题和DeepSeek的思维过程
6. 流式接收并显示Gemini的优化回答
7. 在界面上展示最终结果，并提供查看思维过程的选项
8. 显示统计信息（耗时、token消耗等）

### 关键组件

#### OpenAICompatibleComposite

核心组合器类，负责：
- 初始化并管理DeepSeek和Gemini客户端
- 协调两个模型之间的数据流转
- 处理流式响应的合并和转发
- 统计token使用情况

#### DeepSeekClient

DeepSeek API客户端，负责：
- 处理与DeepSeek API的通信
- 修改用户输入，添加思维链提示
- 流式接收并解析API响应
- 统计token使用情况

#### GeminiClientNew

Gemini API客户端，负责：
- 基于google-genai库与Gemini API通信
- 将OpenAI格式的消息转换为Gemini格式
- 流式接收并解析API响应
- 统计token使用情况

## 🚀 安装步骤

1. 克隆此仓库或下载源代码
   ```
   git clone https://github.com/yifvs/deepgemini.git
   ```

2. 安装依赖包
   ```
   pip install -r requirements.txt
   ```
3、在根目录创建一个'.env'文件，并在文件中添加代理设置
   ```
   http://host:port或socks5://host:port
   ```

## 运行应用

```
streamlit run app.py
```

应用将在本地启动，通常可以通过浏览器访问 http://localhost:8501

## 🎯 使用方法

1. 在应用侧边栏的「API设置」部分输入你的API密钥：
   - DeepSeek API密钥
   - DeepSeek API地址（默认为https://api.deepseek.com/v1/chat/completions，也支持硅基流动）
   - Gemini API密钥

2. 在主界面的输入框中输入你的问题

3. 应用会先使用DeepSeek生成思维链分析

4. 然后用Gemini对分析结果进行总结和优化

5. 你可以点击「查看思维过程」来展开查看DeepSeek的详细分析

6. 每次回答后会显示统计信息，包括耗时和token消耗情况

## 获取API密钥

- Gemini API密钥: 访问 [Google AI Studio](https://makersuite.google.com/app/apikey) 获取
- DeepSeek API密钥: 访问 [DeepSeek官网](https://platform.deepseek.com/) 获取
