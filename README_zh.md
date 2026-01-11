# Toolify

[English](README.md) | [简体中文](README_zh.md)

**为任何大型语言模型赋予函数调用能力。**

Toolify是一个中间件代理，旨在为那些本身不支持函数调用功能的大型语言模型，或未提供函数调用功能的OpenAI接口注入兼容OpenAI格式的函数调用能力。它作为您的应用程序和上游LLM API之间的中介，负责注入必要的提示词并从模型的响应中解析工具调用。

## 核心特性

- **通用函数调用**：为遵循OpenAI API格式但缺乏原生支持的LLM或接口启用函数调用。
- **多函数调用支持**：支持在单次响应中同时执行多个函数。
- **灵活的调用时机**：允许在模型输出的任意阶段启动函数调用。
- **兼容`<think>`标签**：无缝处理`<think>`标签，确保它们不会干扰工具解析。
- **流式响应支持**：全面支持流式响应，实时检测和解析函数调用。
- **多服务路由**：根据请求的模型名称，将请求路由到不同的上游服务。
- **客户端认证**：通过可配置的客户端API密钥保护中间件安全。
- **增强的上下文感知**：在返回工具执行结果时，会基于请求携带的对话历史补充该工具的名称与参数，一并提供给上游模型，以提升上下文理解能力。
- **Token计数**：在响应中提供准确的token使用统计，包括对`reasoning_content`token的支持。
- **自动重试**：当函数调用解析失败时自动提供错误信息并要求模型重试，提高可靠性。

## 工作原理

1. **拦截请求**：Toolify拦截来自客户端的`chat/completions`请求，该请求包含所需的工具定义。
2. **注入提示词**：生成一个特定的系统提示词，指导LLM使用结构化的XML格式和唯一的触发信号来输出函数调用。
3. **代理到上游**：将修改后的请求发送到配置的上游LLM服务。
4. **解析响应**：Toolify分析上游响应。如果检测到触发信号，它会解析XML结构以提取函数调用。
5. **格式化响应**：将解析出的工具调用转换为标准的OpenAI`tool_calls`格式，并将其发送回客户端。

## 安装与设置

您可以通过Docker Compose或使用Python直接运行Toolify。

### 选项 1: 使用 Docker Compose

这是推荐的简易部署方式。

#### 前提条件

- 已安装Docker和Docker Compose。

#### 步骤

1. **克隆仓库：**

   ```bash
   git clone https://github.com/funnycups/toolify.git
   cd toolify
   ```

2. **配置应用程序：**

   复制示例配置文件并进行编辑：

   ```bash
   cp config.example.yaml config.yaml
   ```

   编辑`config.yaml`。`docker-compose.yml`文件已配置为将此文件挂载到容器中。

3. **启动服务：**

   ```bash
   docker-compose up -d
   ```

   这将构建Docker镜像并以后台模式启动Toolify服务，可通过`http://localhost:8000`访问。

### 选项 2: 使用 Python

#### 前提条件

- Python 3.8+

#### 步骤

1. **克隆仓库：**

   ```bash
   git clone https://github.com/funnycups/toolify.git
   cd toolify
   ```

2. **安装依赖：**

   ```bash
   pip install -r requirements.txt
   ```

3. **配置应用程序：**

   复制示例配置文件并进行编辑：

   ```bash
   cp config.example.yaml config.yaml
   ```

   编辑`config.yaml`文件，设置您的上游服务、API密钥以及允许的客户端密钥。

4. **运行服务器：**

   ```bash
   python main.py
   ```

## 配置(`config.yaml`)

请参考[`config.example.yaml`](config.example.yaml)获取详细的配置选项说明。

- **`server`**：中间件的主机、端口和超时设置。
- **`upstream_services`**：上游LLM提供商列表。
  - 定义`base_url`、`api_key`、支持的`models`，并设置一个服务为`is_default: true`。
- **`client_authentication`**：允许访问此中间件的客户端`allowed_keys`列表。
- **`features`**：切换日志记录、角色转换和API密钥处理等功能。
  - `key_passthrough`：设置为`true`时，将直接把客户端提供的API密钥转发给上游服务，而不是使用`upstream_services`中配置的`api_key`。
  - `model_passthrough`：设置为`true`时，将所有请求直接转发到名为'openai'的上游服务，忽略任何基于模型的路由规则。
  - `prompt_template`：自定义用于指导模型如何使用工具的系统提示词。
  - `enable_fc_error_retry`：设置为`true`时，启用函数调用解析失败自动重试功能。
  - `fc_error_retry_max_attempts`：最大重试次数(1-10，默认：3)。

## 使用方法

Toolify运行后，将您的客户端应用程序（例如使用OpenAI SDK）的`base_url`配置为Toolify的地址。使用您配置的`allowed_keys`之一进行身份验证。

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",  # Toolify端点
    api_key="sk-my-secret-key-1"          # 您配置的客户端密钥
)

# 其余的OpenAI API调用保持不变，包括工具定义。
```

Toolify负责处理标准OpenAI工具格式与不支持的LLM所需的基于提示词的方法之间的转换。

## 许可证

本项目采用GPL-3.0-or-later许可证。