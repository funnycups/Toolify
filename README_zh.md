# Toolify

[English](README.md) | [简体中文](README_zh.md)

**为任何大型语言模型赋予函数调用能力。**

Toolify 是一个中间件代理，旨在为那些本身不支持函数调用功能的大型语言模型，或未提供函数调用功能的 OpenAI 接口注入兼容 OpenAI 格式的函数调用能力。它作为您的应用程序和上游 LLM API 之间的中介，负责注入必要的提示词并从模型的响应中解析工具调用。

## 核心特性

- **通用函数调用**：为遵循 OpenAI API 格式但缺乏原生支持的 LLM 或接口启用函数调用。
- **多函数调用支持**：支持在单次响应中同时执行多个函数。
- **灵活的调用时机**：允许在模型输出的任意阶段启动函数调用。
- **兼容 `<think>` 标签**：无缝处理 `<think>` 标签，确保它们不会干扰工具解析。
- **流式响应支持**：全面支持流式响应，实时检测和解析函数调用。
- **多服务路由**：根据请求的模型名称，将请求路由到不同的上游服务。
- **客户端认证**：通过可配置的客户端 API 密钥保护中间件安全。
- **增强的上下文感知**：在返回工具执行结果时，同时向 LLM 提供先前调用的工具详情（名称和参数），提升模型的上下文理解能力。

## 工作原理

1. **拦截请求**：Toolify 拦截来自客户端的 `chat/completions` 请求，该请求包含所需的工具定义。
2. **注入提示词**：生成一个特定的系统提示词，指导 LLM 使用结构化的 XML 格式和唯一的触发信号来输出函数调用。
3. **代理到上游**：将修改后的请求发送到配置的上游 LLM 服务。
4. **解析响应**：Toolify 分析上游响应。如果检测到触发信号，它会解析 XML 结构以提取函数调用。
5. **格式化响应**：将解析出的工具调用转换为标准的 OpenAI `tool_calls` 格式，并将其发送回客户端。

## 安装与设置

您可以使用 Python 直接运行 Toolify，也可以通过 Docker Compose 运行。

### 选项 1: 使用 Python

#### 前提条件

- Python 3.8+

#### 步骤

1. **克隆仓库：**

   ```bash
   git clone https://github.com/yourusername/toolify.git
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

   编辑 `config.yaml` 文件，设置您的上游服务、API 密钥以及允许的客户端密钥。

4. **运行服务器：**

   ```bash
   python main.py
   ```

### 选项 2: 使用 Docker Compose

这是推荐的简易部署方式。

#### 前提条件

- 已安装 Docker 和 Docker Compose。

#### 步骤

1. **克隆仓库：**

   ```bash
   git clone https://github.com/yourusername/toolify.git
   cd toolify
   ```

2. **配置应用程序：**

   复制示例配置文件并进行编辑：

   ```bash
   cp config.example.yaml config.yaml
   ```

   编辑 `config.yaml`。`docker-compose.yml` 文件已配置为将此文件挂载到容器中。

3. **启动服务：**

   ```bash
   docker-compose up -d --build
   ```

   这将构建 Docker 镜像并以后台模式启动 Toolify 服务，可通过 `http://localhost:8000` 访问。

## 配置 (`config.yaml`)

请参考 [`config.example.yaml`](config.example.yaml) 获取详细的配置选项说明。

- **`server`**：中间件的主机、端口和超时设置。
- **`upstream_services`**：上游 LLM 提供商列表。
  - 定义 `base_url`、`api_key`、支持的 `models`，并设置一个服务为 `is_default: true`。
- **`client_authentication`**：允许访问此中间件的客户端 `allowed_keys` 列表。
- **`features`**：切换日志记录、流式传输和角色转换等功能。

## 使用方法

Toolify 运行后，将您的客户端应用程序（例如使用 OpenAI SDK）的 `base_url` 配置为 Toolify 的地址。使用您配置的 `allowed_keys` 之一进行身份验证。

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",  # Toolify 终结点
    api_key="sk-my-secret-key-1"          # 您配置的客户端密钥
)

# 其余的 OpenAI API 调用保持不变，包括工具定义。
```

Toolify 负责处理标准 OpenAI 工具格式与不支持的 LLM 所需的基于提示词的方法之间的转换。

## 许可证

本项目采用 GPL-3.0-or-later 许可证。