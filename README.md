# Toolify

[English](README.md) | [简体中文](README_zh.md)

**Empower any LLM with function calling capabilities.**

Toolify is a middleware proxy designed to inject OpenAI-compatible function calling capabilities into Large Language Models that do not natively support it, or into OpenAI API interfaces that do not provide this functionality. It acts as an intermediary between your application and the upstream LLM API, injecting necessary prompts and parsing tool calls from the model's response.

## Key Features

- **Universal Function Calling**: Enables function calling for LLMs or APIs that adhere to the OpenAI format but lack native support.
- **Multiple Function Calls**: Supports executing multiple functions simultaneously in a single response.
- **Flexible Initiation**: Allows function calls to be initiated at any stage of the model's output.
- **Think Tag Compatibility**: Seamlessly handles `<think>` tags, ensuring they don't interfere with tool parsing.
- **Streaming Support**: Fully supports streaming responses, detecting and parsing function calls on the fly.
- **Multi-Service Routing**: Routes requests to different upstream services based on the requested model name.
- **Client Authentication**: Secures the middleware with configurable client API keys.
- **Enhanced Context Awareness**: Provides LLMs with the details (name and parameters) of previous tool calls alongside the execution results, improving contextual understanding.

## How It Works

1. **Intercept Request**: Toolify intercepts the `chat/completions` request from the client, which includes the desired tools.
2. **Inject Prompt**: It generates a specific system prompt instructing the LLM how to output function calls using a structured XML format and a unique trigger signal.
3. **Proxy to Upstream**: The modified request is sent to the configured upstream LLM service.
4. **Parse Response**: Toolify analyzes the upstream response. If the trigger signal is detected, it parses the XML structure to extract the function calls.
5. **Format Response**: It transforms the parsed tool calls into the standard OpenAI `tool_calls` format and sends it back to the client.

## Installation and Setup

You can run Toolify using Python directly or via Docker Compose.

### Option 1: Using Python

#### Prerequisites

- Python 3.8+

#### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/toolify.git
   cd toolify
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the application:**

   Copy the example configuration file and edit it:

   ```bash
   cp config.example.yaml config.yaml
   ```

   Edit `config.yaml` to set up your upstream services, API keys, and allowed client keys.

4. **Run the server:**

   ```bash
   python main.py
   ```

### Option 2: Using Docker Compose

This is the recommended way for easy deployment.

#### Prerequisites

- Docker and Docker Compose installed.

#### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/toolify.git
   cd toolify
   ```

2. **Configure the application:**

   Copy the example configuration file and edit it:

   ```bash
   cp config.example.yaml config.yaml
   ```

   Edit `config.yaml`. The `docker-compose.yml` file is configured to mount this file into the container.

3. **Start the service:**

   ```bash
   docker-compose up -d --build
   ```

   This will build the Docker image and start the Toolify service in detached mode, accessible at `http://localhost:8000`.

## Configuration (`config.yaml`)

Refer to [`config.example.yaml`](config.example.yaml) for detailed configuration options.

- **`server`**: Middleware host, port, and timeout settings.
- **`upstream_services`**: List of upstream LLM providers (e.g., Groq, OpenAI, Anthropic).
  - Define `base_url`, `api_key`, supported `models`, and set one service as `is_default: true`.
- **`client_authentication`**: List of `allowed_keys` for clients accessing this middleware.
- **`features`**: Toggle features like logging, streaming, and role conversion.

## Usage

Once Toolify is running, configure your client application (e.g., using the OpenAI SDK) to use Toolify's address as the `base_url`. Use one of the configured `allowed_keys` for authentication.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",  # Toolify endpoint
    api_key="sk-my-secret-key-1"          # Your configured client key
)

# The rest of your OpenAI API calls remain the same, including tool definitions.
```

Toolify handles the translation between the standard OpenAI tool format and the prompt-based method required by unsupported LLMs.

## License

This project is licensed under the GPL-3.0-or-later license.