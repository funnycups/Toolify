# SPDX-License-Identifier: GPL-3.0-or-later
#
# Toolify: Empower any LLM with function calling capabilities.
# Copyright (C) 2025 FunnyCups (https://github.com/funnycups)

import os
import re
import json
import uuid
import httpx
import secrets
import string
import traceback
import time
import random
import threading
import logging
import tiktoken
from typing import List, Dict, Any, Optional, Literal, Union
from collections import OrderedDict

from fastapi import FastAPI, Request, Header, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ValidationError

from config_loader import config_loader

logger = logging.getLogger(__name__)

# Token Counter for counting tokens
class TokenCounter:
    """Token counter using tiktoken"""
    
    # Model prefix to encoding mapping (from tiktoken source)
    MODEL_PREFIX_TO_ENCODING = {
        "o1-": "o200k_base",
        "o3-": "o200k_base",
        "o4-mini-": "o200k_base",
        # chat
        "gpt-5-": "o200k_base",
        "gpt-4.5-": "o200k_base",
        "gpt-4.1-": "o200k_base",
        "chatgpt-4o-": "o200k_base",
        "gpt-4o-": "o200k_base",
        "gpt-4-": "cl100k_base",
        "gpt-3.5-turbo-": "cl100k_base",
        "gpt-35-turbo-": "cl100k_base",  # Azure deployment name
        "gpt-oss-": "o200k_harmony",
        # fine-tuned
        "ft:gpt-4o": "o200k_base",
        "ft:gpt-4": "cl100k_base",
        "ft:gpt-3.5-turbo": "cl100k_base",
        "ft:davinci-002": "cl100k_base",
        "ft:babbage-002": "cl100k_base",
    }
    
    def __init__(self):
        self.encoders = {}
    
    def get_encoder(self, model: str):
        """Get or create encoder for the model"""
        if model not in self.encoders:
            encoding = None
            
            # First try to get encoding from model name directly
            try:
                self.encoders[model] = tiktoken.encoding_for_model(model)
                return self.encoders[model]
            except KeyError:
                pass
            
            # Try to find encoding by prefix matching
            for prefix, enc_name in self.MODEL_PREFIX_TO_ENCODING.items():
                if model.startswith(prefix):
                    encoding = enc_name
                    break
            
            # Default to o200k_base for newer models
            if encoding is None:
                logger.warning(f"Model {model} not found in prefix mapping, using o200k_base encoding")
                encoding = "o200k_base"
            
            try:
                self.encoders[model] = tiktoken.get_encoding(encoding)
            except Exception as e:
                logger.warning(f"Failed to get encoding {encoding} for model {model}: {e}. Falling back to cl100k_base")
                self.encoders[model] = tiktoken.get_encoding("cl100k_base")
                
        return self.encoders[model]
    
    def count_tokens(self, messages: list, model: str = "gpt-3.5-turbo") -> int:
        """Count tokens in message list"""
        encoder = self.get_encoder(model)
        
        # All modern chat models use similar token counting
        return self._count_chat_tokens(messages, encoder, model)
    
    def _count_chat_tokens(self, messages: list, encoder, model: str) -> int:
        """Accurate token calculation for chat models
        
        Based on OpenAI's token counting documentation:
        - Each message has a fixed overhead
        - Content tokens are counted per message
        - Special tokens for message formatting
        """
        # Token overhead varies by model
        if model.startswith(("gpt-3.5-turbo", "gpt-35-turbo")):
            # gpt-3.5-turbo uses different message overhead
            tokens_per_message = 4  # <|start|>role<|separator|>content<|end|>
            tokens_per_name = -1    # Name is omitted if not present
        else:
            # Most models including gpt-4, gpt-4o, o1, etc.
            tokens_per_message = 3
            tokens_per_name = 1
        
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            
            # Count tokens for each field in the message
            for key, value in message.items():
                if key == "content":
                    # Handle case where content might be a list (multimodal messages)
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict) and item.get("type") == "text":
                                content_text = item.get("text", "")
                                num_tokens += len(encoder.encode(content_text))
                            # Note: Image tokens are not counted here as they have fixed costs
                    elif isinstance(value, str):
                        num_tokens += len(encoder.encode(value))
                elif key == "name":
                    num_tokens += tokens_per_name
                    if isinstance(value, str):
                        num_tokens += len(encoder.encode(value))
                elif key == "role":
                    # Role is already counted in tokens_per_message
                    pass
                elif isinstance(value, str):
                    # Other string fields
                    num_tokens += len(encoder.encode(value))
        
        # Every reply is primed with assistant role
        num_tokens += 3
        return num_tokens
    
    def count_text_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """Count tokens in plain text"""
        encoder = self.get_encoder(model)
        return len(encoder.encode(text))

# Global token counter instance
token_counter = TokenCounter()

def generate_random_trigger_signal() -> str:
    """Generate a random, self-closing trigger signal like <Function_AB1c_Start/>."""
    chars = string.ascii_letters + string.digits
    random_str = ''.join(secrets.choice(chars) for _ in range(4))
    return f"<Function_{random_str}_Start/>"

try:
    app_config = config_loader.load_config()
    
    log_level_str = app_config.features.log_level
    if log_level_str == "DISABLED":
        log_level = logging.CRITICAL + 1
    else:
        log_level = getattr(logging, log_level_str, logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger.info(f"✅ Configuration loaded successfully: {config_loader.config_path}")
    logger.info(f"📊 Configured {len(app_config.upstream_services)} upstream services")
    logger.info(f"🔑 Configured {len(app_config.client_authentication.allowed_keys)} client keys")
    
    MODEL_TO_SERVICE_MAPPING, ALIAS_MAPPING = config_loader.get_model_to_service_mapping()
    DEFAULT_SERVICE = config_loader.get_default_service()
    ALLOWED_CLIENT_KEYS = config_loader.get_allowed_client_keys()
    GLOBAL_TRIGGER_SIGNAL = generate_random_trigger_signal()
    
    logger.info(f"🎯 Configured {len(MODEL_TO_SERVICE_MAPPING)} model mappings")
    if ALIAS_MAPPING:
        logger.info(f"🔄 Configured {len(ALIAS_MAPPING)} model aliases: {list(ALIAS_MAPPING.keys())}")
    logger.info(f"🔄 Default service: {DEFAULT_SERVICE['name']}")
    
except Exception as e:
    logger.error(f"❌ Configuration loading failed: {type(e).__name__}")
    logger.error(f"❌ Error details: {str(e)}")
    logger.error("💡 Please ensure config.yaml file exists and is properly formatted")
    exit(1)
class ToolCallMappingManager:
    """
    Tool call mapping manager with TTL (Time To Live) and size limit
    
    Features:
    1. Automatic expiration cleanup - entries are automatically deleted after specified time
    2. Size limit - prevents unlimited memory growth
    3. LRU eviction - removes least recently used entries when size limit is reached
    4. Thread safe - supports concurrent access
    5. Periodic cleanup - background thread regularly cleans up expired entries
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600, cleanup_interval: int = 300):
        """
        Initialize mapping manager
        
        Args:
            max_size: Maximum number of stored entries
            ttl_seconds: Entry time to live (seconds)
            cleanup_interval: Cleanup interval (seconds)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval
        
        self._data: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
        
        logger.debug(f"🔧 [INIT] Tool call mapping manager started - Max entries: {max_size}, TTL: {ttl_seconds}s, Cleanup interval: {cleanup_interval}s")
    
    def store(self, tool_call_id: str, name: str, args: dict, description: str = "") -> None:
        """Store tool call mapping"""
        with self._lock:
            current_time = time.time()
            
            if tool_call_id in self._data:
                del self._data[tool_call_id]
                del self._timestamps[tool_call_id]
            
            while len(self._data) >= self.max_size:
                oldest_key = next(iter(self._data))
                del self._data[oldest_key]
                del self._timestamps[oldest_key]
                logger.debug(f"🔧 [CLEANUP] Removed oldest entry due to size limit: {oldest_key}")
            
            self._data[tool_call_id] = {
                "name": name,
                "args": args,
                "description": description,
                "created_at": current_time
            }
            self._timestamps[tool_call_id] = current_time
            
            logger.debug(f"🔧 Stored tool call mapping: {tool_call_id} -> {name}")
            logger.debug(f"🔧 Current mapping table size: {len(self._data)}")
    
    def get(self, tool_call_id: str) -> Optional[Dict[str, Any]]:
        """Get tool call mapping (updates LRU order)"""
        with self._lock:
            current_time = time.time()
            
            if tool_call_id not in self._data:
                logger.debug(f"🔧 Tool call mapping not found: {tool_call_id}")
                logger.debug(f"🔧 All IDs in current mapping table: {list(self._data.keys())}")
                return None
            
            if current_time - self._timestamps[tool_call_id] > self.ttl_seconds:
                logger.debug(f"🔧 Tool call mapping expired: {tool_call_id}")
                del self._data[tool_call_id]
                del self._timestamps[tool_call_id]
                return None
            
            result = self._data[tool_call_id]
            self._data.move_to_end(tool_call_id)
            
            logger.debug(f"🔧 Found tool call mapping: {tool_call_id} -> {result['name']}")
            return result
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries, return the number of cleaned entries"""
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, timestamp in self._timestamps.items():
                if current_time - timestamp > self.ttl_seconds:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._data[key]
                del self._timestamps[key]
            
            if expired_keys:
                logger.debug(f"🔧 [CLEANUP] Cleaned up {len(expired_keys)} expired entries")
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        with self._lock:
            current_time = time.time()
            expired_count = sum(1 for ts in self._timestamps.values()
                              if current_time - ts > self.ttl_seconds)
            
            return {
                "total_entries": len(self._data),
                "expired_entries": expired_count,
                "active_entries": len(self._data) - expired_count,
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "memory_usage_ratio": len(self._data) / self.max_size
            }
    
    def _periodic_cleanup(self) -> None:
        """Background periodic cleanup thread"""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                cleaned = self.cleanup_expired()
                
                stats = self.get_stats()
                if stats["total_entries"] > 0:
                    logger.debug(f"🔧 [STATS] Mapping table status: Total={stats['total_entries']}, "
                               f"Active={stats['active_entries']}, Memory usage={stats['memory_usage_ratio']:.1%}")
                
            except Exception as e:
                logger.error(f"❌ Background cleanup thread exception: {e}")

TOOL_CALL_MAPPING_MANAGER = ToolCallMappingManager(
    max_size=1000,
    ttl_seconds=3600,
    cleanup_interval=300
)

def store_tool_call_mapping(tool_call_id: str, name: str, args: dict, description: str = ""):
    """Store mapping between tool call ID and call content"""
    TOOL_CALL_MAPPING_MANAGER.store(tool_call_id, name, args, description)

def get_tool_call_mapping(tool_call_id: str) -> Optional[Dict[str, Any]]:
    """Get call content corresponding to tool call ID"""
    return TOOL_CALL_MAPPING_MANAGER.get(tool_call_id)

def format_tool_result_for_ai(tool_call_id: str, result_content: str) -> str:
    """Format tool call results for AI understanding with English prompts and XML structure"""
    logger.debug(f"🔧 Formatting tool call result: tool_call_id={tool_call_id}")
    tool_info = get_tool_call_mapping(tool_call_id)
    if not tool_info:
        logger.debug(f"🔧 Tool call mapping not found, using default format")
        return f"Tool execution result:\n<tool_result>\n{result_content}\n</tool_result>"
    
    formatted_text = f"""Tool execution result:
- Tool name: {tool_info['name']}
- Execution result:
<tool_result>
{result_content}
</tool_result>"""
    
    logger.debug(f"🔧 Formatting completed, tool name: {tool_info['name']}")
    return formatted_text

def format_assistant_tool_calls_for_ai(tool_calls: List[Dict[str, Any]], trigger_signal: str) -> str:
    """Format assistant tool calls into AI-readable string format."""
    logger.debug(f"🔧 Formatting assistant tool calls. Count: {len(tool_calls)}")
    
    xml_calls_parts = []
    for tool_call in tool_calls:
        function_info = tool_call.get("function", {})
        name = function_info.get("name", "")
        arguments_json = function_info.get("arguments", "{}")
        
        try:
            # First, try to load as JSON. If it's a string that's a valid JSON, we parse it.
            args_dict = json.loads(arguments_json)
        except (json.JSONDecodeError, TypeError):
            # If it's not a valid JSON string, treat it as a simple string.
            args_dict = {"raw_arguments": arguments_json}

        args_parts = []
        for key, value in args_dict.items():
            # Dump the value back to a JSON string for consistent representation inside XML.
            json_value = json.dumps(value, ensure_ascii=False)
            args_parts.append(f"<{key}>{json_value}</{key}>")
        
        args_content = "\n".join(args_parts)
        
        xml_call = f"<function_call>\n<tool>{name}</tool>\n<args>\n{args_content}\n</args>\n</function_call>"
        xml_calls_parts.append(xml_call)

    all_calls = "\n".join(xml_calls_parts)
    final_str = f"{trigger_signal}\n<function_calls>\n{all_calls}\n</function_calls>"
    
    logger.debug("🔧 Assistant tool calls formatted successfully.")
    return final_str

def get_function_call_prompt_template(trigger_signal: str) -> str:
    """
    Generate prompt template based on dynamic trigger signal
    """
    custom_template = app_config.features.prompt_template
    if custom_template:
        logger.info("🔧 Using custom prompt template from configuration")
        return custom_template.format(
            trigger_signal=trigger_signal,
            tools_list="{tools_list}"
        )
    
    return f"""
You have access to the following available tools to help solve problems:

{{tools_list}}

**IMPORTANT CONTEXT NOTES:**
1. You can call MULTIPLE tools in a single response if needed.
2. The conversation context may already contain tool execution results from previous function calls. Review the conversation history carefully to avoid unnecessary duplicate tool calls.
3. When tool execution results are present in the context, they will be formatted with XML tags like <tool_result>...</tool_result> for easy identification.
4. This is the ONLY format you can use for tool calls, and any deviation will result in failure.

When you need to use tools, you **MUST** strictly follow this format. Do NOT include any extra text, explanations, or dialogue on the first and second lines of the tool call syntax:

1. When starting tool calls, begin on a new line with exactly:
{trigger_signal}
No leading or trailing spaces, output exactly as shown above. The trigger signal MUST be on its own line and appear only once.

2. Starting from the second line, **immediately** follow with the complete <function_calls> XML block.

3. For multiple tool calls, include multiple <function_call> blocks within the same <function_calls> wrapper.

4. Do not add any text or explanation after the closing </function_calls> tag.

STRICT ARGUMENT KEY RULES:
- You MUST use parameter keys EXACTLY as defined (case- and punctuation-sensitive). Do NOT rename, add, or remove characters.
- If a key starts with a hyphen (e.g., -i, -C), you MUST keep the hyphen in the tag name. Example: <-i>true</-i>, <-C>2</-C>.
- Never convert "-i" to "i" or "-C" to "C". Do not pluralize, translate, or alias parameter keys.
- The <tool> tag must contain the exact name of a tool from the list. Any other tool name is invalid.
- The <args> must contain all required arguments for that tool.

CORRECT Example (multiple tool calls, including hyphenated keys):
...response content (optional)...
{trigger_signal}
<function_calls>
    <function_call>
        <tool>Grep</tool>
        <args>
            <-i>true</-i>
            <-C>2</-C>
            <path>.</path>
        </args>
    </function_call>
    <function_call>
        <tool>search</tool>
        <args>
            <keywords>["Python Document", "how to use python"]</keywords>
        </args>
    </function_call>
  </function_calls>

INCORRECT Example (extra text + wrong key names — DO NOT DO THIS):
...response content (optional)...
{trigger_signal}
I will call the tools for you.
<function_calls>
    <function_call>
        <tool>Grep</tool>
        <args>
            <i>true</i>
            <C>2</C>
            <path>.</path>
        </args>
    </function_call>
</function_calls>

Now please be ready to strictly follow the above specifications.
"""

class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]

class Tool(BaseModel):
    type: Literal["function"]
    function: ToolFunction

class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    
    class Config:
        extra = "allow"

class ToolChoice(BaseModel):
    type: Literal["function"]
    function: Dict[str, str]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, ToolChoice]] = None
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    
    class Config:
        extra = "allow"


def generate_function_prompt(tools: List[Tool], trigger_signal: str) -> tuple[str, str]:
    """
    Generate injected system prompt based on tools definition in client request.
    Returns: (prompt_content, trigger_signal)
    """
    tools_list_str = []
    for i, tool in enumerate(tools):
        func = tool.function
        name = func.name
        description = func.description or ""

        # Robustly read JSON Schema fields
        schema: Dict[str, Any] = func.parameters or {}
        props: Dict[str, Any] = schema.get("properties", {}) or {}
        required_list: List[str] = schema.get("required", []) or []

        # Brief summary line: name (type)
        params_summary = ", ".join([
            f"{p_name} ({(p_info or {}).get('type', 'any')})" for p_name, p_info in props.items()
        ]) or "None"

        # Build detailed parameter spec for prompt injection (default enabled)
        detail_lines: List[str] = []
        for p_name, p_info in props.items():
            p_info = p_info or {}
            p_type = p_info.get("type", "any")
            is_required = "Yes" if p_name in required_list else "No"
            p_desc = p_info.get("description")
            enum_vals = p_info.get("enum")
            default_val = p_info.get("default")
            examples_val = p_info.get("examples") or p_info.get("example")

            # Common constraints and hints
            constraints: Dict[str, Any] = {}
            for key in [
                "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                "minLength", "maxLength", "pattern", "format",
                "minItems", "maxItems", "uniqueItems"
            ]:
                if key in p_info:
                    constraints[key] = p_info.get(key)

            # Array item type hint
            if p_type == "array":
                items = p_info.get("items") or {}
                if isinstance(items, dict):
                    itype = items.get("type")
                    if itype:
                        constraints["items.type"] = itype

            # Compose pretty lines
            detail_lines.append(f"- {p_name}:")
            detail_lines.append(f"  - type: {p_type}")
            detail_lines.append(f"  - required: {is_required}")
            if p_desc:
                detail_lines.append(f"  - description: {p_desc}")
            if enum_vals is not None:
                try:
                    detail_lines.append(f"  - enum: {json.dumps(enum_vals, ensure_ascii=False)}")
                except Exception:
                    detail_lines.append(f"  - enum: {enum_vals}")
            if default_val is not None:
                try:
                    detail_lines.append(f"  - default: {json.dumps(default_val, ensure_ascii=False)}")
                except Exception:
                    detail_lines.append(f"  - default: {default_val}")
            if examples_val is not None:
                try:
                    detail_lines.append(f"  - examples: {json.dumps(examples_val, ensure_ascii=False)}")
                except Exception:
                    detail_lines.append(f"  - examples: {examples_val}")
            if constraints:
                try:
                    detail_lines.append(f"  - constraints: {json.dumps(constraints, ensure_ascii=False)}")
                except Exception:
                    detail_lines.append(f"  - constraints: {constraints}")

        detail_block = "\n".join(detail_lines) if detail_lines else "(no parameter details)"

        desc_block = f"```\n{description}\n```" if description else "None"

        tools_list_str.append(
            f"{i + 1}. <tool name=\"{name}\">\n"
            f"   Description:\n{desc_block}\n"
            f"   Parameters summary: {params_summary}\n"
            f"   Required parameters: {', '.join(required_list) if required_list else 'None'}\n"
            f"   Parameter details:\n{detail_block}"
        )
    
    prompt_template = get_function_call_prompt_template(trigger_signal)
    prompt_content = prompt_template.replace("{tools_list}", "\n\n".join(tools_list_str))
    
    return prompt_content, trigger_signal

def remove_think_blocks(text: str) -> str:
    """
    Temporarily remove all <think>...</think> blocks for XML parsing
    Supports nested think tags
    Note: This function is only used for temporary parsing and does not affect the original content returned to the user
    """
    while '<think>' in text and '</think>' in text:
        start_pos = text.find('<think>')
        if start_pos == -1:
            break
        
        pos = start_pos + 7
        depth = 1
        
        while pos < len(text) and depth > 0:
            if text[pos:pos+7] == '<think>':
                depth += 1
                pos += 7
            elif text[pos:pos+8] == '</think>':
                depth -= 1
                pos += 8
            else:
                pos += 1
        
        if depth == 0:
            text = text[:start_pos] + text[pos:]
        else:
            break
    
    return text

class StreamingFunctionCallDetector:
    """Enhanced streaming function call detector, supports dynamic trigger signals, avoids misjudgment within <think> tags
    
    Core features:
    1. Avoid triggering tool call detection within <think> blocks
    2. Normally output <think> block content to the user
    3. Supports nested think tags
    """
    
    def __init__(self, trigger_signal: str):
        self.trigger_signal = trigger_signal
        self.reset()
    
    def reset(self):
        self.content_buffer = ""
        self.state = "detecting"  # detecting, tool_parsing
        self.in_think_block = False
        self.think_depth = 0
        self.signal = self.trigger_signal
        self.signal_len = len(self.signal)
    
    def process_chunk(self, delta_content: str) -> tuple[bool, str]:
        """
        Process streaming content chunk
        Returns: (is_tool_call_detected, content_to_yield)
        """
        if not delta_content:
            return False, ""
        
        self.content_buffer += delta_content
        content_to_yield = ""
        
        if self.state == "tool_parsing":
            return False, ""
        
        if delta_content:
            logger.debug(f"🔧 Processing chunk: {repr(delta_content[:50])}{'...' if len(delta_content) > 50 else ''}, buffer length: {len(self.content_buffer)}, think state: {self.in_think_block}")
        
        i = 0
        while i < len(self.content_buffer):
            skip_chars = self._update_think_state(i)
            if skip_chars > 0:
                for j in range(skip_chars):
                    if i + j < len(self.content_buffer):
                        content_to_yield += self.content_buffer[i + j]
                i += skip_chars
                continue
            
            if not self.in_think_block and self._can_detect_signal_at(i):
                if self.content_buffer[i:i+self.signal_len] == self.signal:
                    logger.debug(f"🔧 Improved detector: detected trigger signal in non-think block! Signal: {self.signal[:20]}...")
                    logger.debug(f"🔧 Trigger signal position: {i}, think state: {self.in_think_block}, think depth: {self.think_depth}")
                    self.state = "tool_parsing"
                    self.content_buffer = self.content_buffer[i:]
                    return True, content_to_yield
            
            remaining_len = len(self.content_buffer) - i
            if remaining_len < self.signal_len or remaining_len < 8:
                break
            
            content_to_yield += self.content_buffer[i]
            i += 1
        
        self.content_buffer = self.content_buffer[i:]
        return False, content_to_yield
    
    def _update_think_state(self, pos: int):
        """Update think tag state, supports nesting"""
        remaining = self.content_buffer[pos:]
        
        if remaining.startswith('<think>'):
            self.think_depth += 1
            self.in_think_block = True
            logger.debug(f"🔧 Entering think block, depth: {self.think_depth}")
            return 7
        
        elif remaining.startswith('</think>'):
            self.think_depth = max(0, self.think_depth - 1)
            self.in_think_block = self.think_depth > 0
            logger.debug(f"🔧 Exiting think block, depth: {self.think_depth}")
            return 8
        
        return 0
    
    def _can_detect_signal_at(self, pos: int) -> bool:
        """Check if signal can be detected at the specified position"""
        return (pos + self.signal_len <= len(self.content_buffer) and 
                not self.in_think_block)
    
    def finalize(self) -> Optional[List[Dict[str, Any]]]:
        """Final processing when stream ends"""
        if self.state == "tool_parsing":
            return parse_function_calls_xml(self.content_buffer, self.trigger_signal)
        return None

def parse_function_calls_xml(xml_string: str, trigger_signal: str) -> Optional[List[Dict[str, Any]]]:
    """
    Enhanced XML parsing function, supports dynamic trigger signals
    1. Retain <think>...</think> blocks (they should be returned normally to the user)
    2. Temporarily remove think blocks only when parsing function_calls to prevent think content from interfering with XML parsing
    3. Find the last occurrence of the trigger signal
    4. Start parsing function_calls from the last trigger signal
    """
    logger.debug(f"🔧 Improved parser starting processing, input length: {len(xml_string) if xml_string else 0}")
    logger.debug(f"🔧 Using trigger signal: {trigger_signal[:20]}...")
    
    if not xml_string or trigger_signal not in xml_string:
        logger.debug(f"🔧 Input is empty or doesn't contain trigger signal")
        return None
    
    cleaned_content = remove_think_blocks(xml_string)
    logger.debug(f"🔧 Content length after temporarily removing think blocks: {len(cleaned_content)}")
    
    signal_positions = []
    start_pos = 0
    while True:
        pos = cleaned_content.find(trigger_signal, start_pos)
        if pos == -1:
            break
        signal_positions.append(pos)
        start_pos = pos + 1
    
    if not signal_positions:
        logger.debug(f"🔧 No trigger signal found in cleaned content")
        return None
    
    logger.debug(f"🔧 Found {len(signal_positions)} trigger signal positions: {signal_positions}")
    
    last_signal_pos = signal_positions[-1]
    content_after_signal = cleaned_content[last_signal_pos:]
    logger.debug(f"🔧 Content starting from last trigger signal: {repr(content_after_signal[:100])}")
    
    calls_content_match = re.search(r"<function_calls>([\s\S]*?)</function_calls>", content_after_signal)
    if not calls_content_match:
        logger.debug(f"🔧 No function_calls tag found")
        return None
    
    calls_content = calls_content_match.group(1)
    logger.debug(f"🔧 function_calls content: {repr(calls_content)}")
    
    results = []
    call_blocks = re.findall(r"<function_call>([\s\S]*?)</function_call>", calls_content)
    logger.debug(f"🔧 Found {len(call_blocks)} function_call blocks")
    
    for i, block in enumerate(call_blocks):
        logger.debug(f"🔧 Processing function_call #{i+1}: {repr(block)}")
        
        tool_match = re.search(r"<tool>(.*?)</tool>", block)
        if not tool_match:
            logger.debug(f"🔧 No tool tag found in block #{i+1}")
            continue
        
        name = tool_match.group(1).strip()
        args = {}
        
        args_block_match = re.search(r"<args>([\s\S]*?)</args>", block)
        if args_block_match:
            args_content = args_block_match.group(1)
            # Support arg tag names containing hyphens (e.g., -i, -A); match any non-space, non-'>' and non-'/' chars
            arg_matches = re.findall(r"<([^\s>/]+)>([\s\S]*?)</\1>", args_content)

            def _coerce_value(v: str):
                try:
                    return json.loads(v)
                except Exception:
                    pass
                return v

            for k, v in arg_matches:
                args[k] = _coerce_value(v)
        
        result = {"name": name, "args": args}
        results.append(result)
        logger.debug(f"🔧 Added tool call: {result}")
    
    logger.debug(f"🔧 Final parsing result: {results}")
    return results if results else None

def find_upstream(model_name: str) -> tuple[Dict[str, Any], str]:
    """Find upstream configuration by model name, handling aliases and passthrough mode."""
    
    # Handle model passthrough mode
    if app_config.features.model_passthrough:
        logger.info("🔄 Model passthrough mode is active. Forwarding to 'openai' service.")
        openai_service = None
        for service in app_config.upstream_services:
            if service.name == "openai":
                openai_service = service.model_dump()
                break
        
        if openai_service:
            if not openai_service.get("api_key"):
                 raise HTTPException(status_code=500, detail="Configuration error: API key not found for the 'openai' service in model passthrough mode.")
            # In passthrough mode, the model name from the request is used directly.
            return openai_service, model_name
        else:
            raise HTTPException(status_code=500, detail="Configuration error: 'model_passthrough' is enabled, but no upstream service named 'openai' was found.")

    # Default routing logic
    chosen_model_entry = model_name
    
    if model_name in ALIAS_MAPPING:
        chosen_model_entry = random.choice(ALIAS_MAPPING[model_name])
        logger.info(f"🔄 Model alias '{model_name}' detected. Randomly selected '{chosen_model_entry}' for this request.")

    service = MODEL_TO_SERVICE_MAPPING.get(chosen_model_entry)
    
    if service:
        if not service.get("api_key"):
            raise HTTPException(status_code=500, detail=f"Model configuration error: API key not found for service '{service.get('name')}'.")
    else:
        logger.warning(f"⚠️  Model '{model_name}' not found in configuration, using default service")
        service = DEFAULT_SERVICE
        if not service.get("api_key"):
            raise HTTPException(status_code=500, detail="Service configuration error: Default API key not found.")

    actual_model_name = chosen_model_entry
    if ':' in chosen_model_entry:
         parts = chosen_model_entry.split(':', 1)
         if len(parts) == 2:
             _, actual_model_name = parts
            
    return service, actual_model_name

app = FastAPI()
http_client = httpx.AsyncClient()

@app.middleware("http")
async def debug_middleware(request: Request, call_next):
    """Middleware for debugging validation errors, does not log conversation content."""
    response = await call_next(request)
    
    if response.status_code == 422:
        logger.debug(f"🔍 Validation error detected for {request.method} {request.url.path}")
        logger.debug(f"🔍 Response status code: 422 (Pydantic validation failure)")
    
    return response

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors with detailed error information"""
    logger.error(f"❌ Pydantic validation error: {exc}")
    logger.error(f"❌ Request URL: {request.url}")
    logger.error(f"❌ Error details: {exc.errors()}")
    
    for error in exc.errors():
        logger.error(f"❌ Validation error location: {error.get('loc')}")
        logger.error(f"❌ Validation error message: {error.get('msg')}")
        logger.error(f"❌ Validation error type: {error.get('type')}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "Invalid request format",
                "type": "invalid_request_error",
                "code": "invalid_request"
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions"""
    logger.error(f"❌ Unhandled exception: {exc}")
    logger.error(f"❌ Request URL: {request.url}")
    logger.error(f"❌ Exception type: {type(exc).__name__}")
    logger.error(f"❌ Error stack: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "code": "internal_error"
            }
        }
    )

async def verify_api_key(authorization: str = Header(...)):
    """Dependency: verify client API key"""
    client_key = authorization.replace("Bearer ", "")
    if app_config.features.key_passthrough:
        # In passthrough mode, skip allowed_keys check
        return client_key
    if client_key not in ALLOWED_CLIENT_KEYS:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return client_key

def preprocess_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Preprocess messages, convert tool-type messages to AI-understandable format, return dictionary list to avoid Pydantic validation issues"""
    processed_messages = []
    
    for message in messages:
        if isinstance(message, dict):
            if message.get("role") == "tool":
                tool_call_id = message.get("tool_call_id")
                content = message.get("content")
                
                if tool_call_id and content:
                    formatted_content = format_tool_result_for_ai(tool_call_id, content)
                    processed_message = {
                        "role": "user",
                        "content": formatted_content
                    }
                    processed_messages.append(processed_message)
                    logger.debug(f"🔧 Converted tool message to user message: tool_call_id={tool_call_id}")
                else:
                    logger.debug(f"🔧 Skipped invalid tool message: tool_call_id={tool_call_id}, content={bool(content)}")
            elif message.get("role") == "assistant" and "tool_calls" in message and message["tool_calls"]:
                tool_calls = message.get("tool_calls", [])
                formatted_tool_calls_str = format_assistant_tool_calls_for_ai(tool_calls, GLOBAL_TRIGGER_SIGNAL)
                
                # Combine with original content if it exists
                original_content = message.get("content") or ""
                final_content = f"{original_content}\n{formatted_tool_calls_str}".strip()

                processed_message = {
                    "role": "assistant",
                    "content": final_content
                }
                # Copy other potential keys from the original message, except tool_calls
                for key, value in message.items():
                    if key not in ["role", "content", "tool_calls"]:
                        processed_message[key] = value

                processed_messages.append(processed_message)
                logger.debug(f"🔧 Converted assistant tool_calls to content.")

            elif message.get("role") == "developer":
                if app_config.features.convert_developer_to_system:
                    processed_message = message.copy()
                    processed_message["role"] = "system"
                    processed_messages.append(processed_message)
                    logger.debug(f"🔧 Converted developer message to system message for better upstream compatibility")
                else:
                    processed_messages.append(message)
                    logger.debug(f"🔧 Keeping developer role unchanged (based on configuration)")
            else:
                processed_messages.append(message)
        else:
            processed_messages.append(message)
    
    return processed_messages

@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    _api_key: str = Depends(verify_api_key)
):
    """Main chat completion endpoint, proxy and inject function calling capabilities."""
    start_time = time.time()
    
    # Count input tokens
    prompt_tokens = token_counter.count_tokens(body.messages, body.model)
    logger.info(f"📊 Request to {body.model} - Input tokens: {prompt_tokens}")
    
    try:
        logger.debug(f"🔧 Received request, model: {body.model}")
        logger.debug(f"🔧 Number of messages: {len(body.messages)}")
        logger.debug(f"🔧 Number of tools: {len(body.tools) if body.tools else 0}")
        logger.debug(f"🔧 Streaming: {body.stream}")
        
        upstream, actual_model = find_upstream(body.model)
        upstream_url = f"{upstream['base_url']}/chat/completions"
        
        logger.debug(f"🔧 Starting message preprocessing, original message count: {len(body.messages)}")
        processed_messages = preprocess_messages(body.messages)
        logger.debug(f"🔧 Preprocessing completed, processed message count: {len(processed_messages)}")
        
        if not validate_message_structure(processed_messages):
            logger.error(f"❌ Message structure validation failed, but continuing processing")
        
        request_body_dict = body.model_dump(exclude_unset=True)
        request_body_dict["model"] = actual_model
        request_body_dict["messages"] = processed_messages
        is_fc_enabled = app_config.features.enable_function_calling
        has_tools_in_request = bool(body.tools)
        has_function_call = is_fc_enabled and has_tools_in_request
        
        logger.debug(f"🔧 Request body constructed, message count: {len(processed_messages)}")
        
    except Exception as e:
        logger.error(f"❌ Request preprocessing failed: {str(e)}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        if hasattr(app_config, 'debug') and app_config.debug:
            logger.error(f"❌ Error stack: {traceback.format_exc()}")
        
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": "Invalid request format",
                    "type": "invalid_request_error",
                    "code": "invalid_request"
                }
            }
        )

    if has_function_call:
        logger.debug(f"🔧 Using global trigger signal for this request: {GLOBAL_TRIGGER_SIGNAL}")
        
        function_prompt, _ = generate_function_prompt(body.tools, GLOBAL_TRIGGER_SIGNAL)
        
        tool_choice_prompt = safe_process_tool_choice(body.tool_choice)
        if tool_choice_prompt:
            function_prompt += tool_choice_prompt

        system_message = {"role": "system", "content": function_prompt}
        request_body_dict["messages"].insert(0, system_message)
        
        if "tools" in request_body_dict:
            del request_body_dict["tools"]
        if "tool_choice" in request_body_dict:
            del request_body_dict["tool_choice"]

    elif has_tools_in_request and not is_fc_enabled:
        logger.info(f"🔧 Function calling is disabled by configuration, ignoring 'tools' and 'tool_choice' in request.")
        if "tools" in request_body_dict:
            del request_body_dict["tools"]
        if "tool_choice" in request_body_dict:
            del request_body_dict["tool_choice"]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_api_key}" if app_config.features.key_passthrough else f"Bearer {upstream['api_key']}",
        "Accept": "application/json" if not body.stream else "text/event-stream"
    }

    logger.info(f"📝 Forwarding request to upstream: {upstream['name']}")
    logger.info(f"📝 Model: {request_body_dict.get('model', 'unknown')}, Messages: {len(request_body_dict.get('messages', []))}")

    if not body.stream:
        try:
            logger.debug(f"🔧 Sending upstream request to: {upstream_url}")
            logger.debug(f"🔧 has_function_call: {has_function_call}")
            logger.debug(f"🔧 Request body contains tools: {bool(body.tools)}")
            
            upstream_response = await http_client.post(
                upstream_url, json=request_body_dict, headers=headers, timeout=app_config.server.timeout
            )
            upstream_response.raise_for_status() # If status code is 4xx or 5xx, raise exception
            
            response_json = upstream_response.json()
            logger.debug(f"🔧 Upstream response status code: {upstream_response.status_code}")
            
            # Count output tokens and handle usage
            completion_text = ""
            if response_json.get("choices") and len(response_json["choices"]) > 0:
                content = response_json["choices"][0].get("message", {}).get("content")
                if content:
                    completion_text = content
            
            # Calculate our estimated tokens
            estimated_completion_tokens = token_counter.count_text_tokens(completion_text, body.model) if completion_text else 0
            estimated_prompt_tokens = prompt_tokens
            estimated_total_tokens = estimated_prompt_tokens + estimated_completion_tokens
            elapsed_time = time.time() - start_time
            
            # Check if upstream provided usage and respect it
            upstream_usage = response_json.get("usage", {})
            if upstream_usage:
                # Preserve upstream's usage structure and only replace zero values
                final_usage = upstream_usage.copy()
                
                # Replace zero or missing values with our estimates
                if not final_usage.get("prompt_tokens") or final_usage.get("prompt_tokens") == 0:
                    final_usage["prompt_tokens"] = estimated_prompt_tokens
                    logger.debug(f"🔧 Replaced zero/missing prompt_tokens with estimate: {estimated_prompt_tokens}")
                
                if not final_usage.get("completion_tokens") or final_usage.get("completion_tokens") == 0:
                    final_usage["completion_tokens"] = estimated_completion_tokens
                    logger.debug(f"🔧 Replaced zero/missing completion_tokens with estimate: {estimated_completion_tokens}")
                
                if not final_usage.get("total_tokens") or final_usage.get("total_tokens") == 0:
                    final_usage["total_tokens"] = final_usage.get("prompt_tokens", estimated_prompt_tokens) + final_usage.get("completion_tokens", estimated_completion_tokens)
                    logger.debug(f"🔧 Replaced zero/missing total_tokens with calculated value: {final_usage['total_tokens']}")
                
                response_json["usage"] = final_usage
                logger.debug(f"🔧 Preserved upstream usage with replacements: {final_usage}")
            else:
                # No upstream usage, provide our estimates
                response_json["usage"] = {
                    "prompt_tokens": estimated_prompt_tokens,
                    "completion_tokens": estimated_completion_tokens,
                    "total_tokens": estimated_total_tokens
                }
                logger.debug(f"🔧 No upstream usage found, using estimates")
            
            # Log token statistics
            actual_usage = response_json["usage"]
            logger.info("=" * 60)
            logger.info(f"📊 Token Usage Statistics - Model: {body.model}")
            logger.info(f"   Input Tokens: {actual_usage.get('prompt_tokens', 0)}")
            logger.info(f"   Output Tokens: {actual_usage.get('completion_tokens', 0)}")
            logger.info(f"   Total Tokens: {actual_usage.get('total_tokens', 0)}")
            logger.info(f"   Duration: {elapsed_time:.2f}s")
            logger.info("=" * 60)
            
            if has_function_call:
                content = response_json["choices"][0]["message"]["content"]
                logger.debug(f"🔧 Complete response content: {repr(content)}")
                
                parsed_tools = parse_function_calls_xml(content, GLOBAL_TRIGGER_SIGNAL)
                logger.debug(f"🔧 XML parsing result: {parsed_tools}")
                
                if parsed_tools:
                    logger.debug(f"🔧 Successfully parsed {len(parsed_tools)} tool calls")
                    tool_calls = []
                    for tool in parsed_tools:
                        tool_call_id = f"call_{uuid.uuid4().hex}"
                        store_tool_call_mapping(
                            tool_call_id,
                            tool["name"],
                            tool["args"],
                            f"Calling tool {tool['name']}"
                        )
                        tool_calls.append({
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool["name"],
                                "arguments": json.dumps(tool["args"])
                            }
                        })
                    logger.debug(f"🔧 Converted tool_calls: {tool_calls}")
                    
                    response_json["choices"][0]["message"] = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls,
                    }
                    response_json["choices"][0]["finish_reason"] = "tool_calls"
                    logger.debug(f"🔧 Function call conversion completed")
                else:
                    logger.debug(f"🔧 No tool calls detected, returning original content (including think blocks)")
            else:
                logger.debug(f"🔧 No function calls detected or conversion conditions not met")
            
            return JSONResponse(content=response_json)

        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Upstream service response error: status_code={e.response.status_code}")
            logger.error(f"❌ Upstream error details: {e.response.text}")
            
            if e.response.status_code == 400:
                error_response = {
                    "error": {
                        "message": "Invalid request parameters",
                        "type": "invalid_request_error",
                        "code": "bad_request"
                    }
                }
            elif e.response.status_code == 401:
                error_response = {
                    "error": {
                        "message": "Authentication failed",
                        "type": "authentication_error", 
                        "code": "unauthorized"
                    }
                }
            elif e.response.status_code == 403:
                error_response = {
                    "error": {
                        "message": "Access forbidden",
                        "type": "permission_error",
                        "code": "forbidden"
                    }
                }
            elif e.response.status_code == 429:
                error_response = {
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded"
                    }
                }
            elif e.response.status_code >= 500:
                error_response = {
                    "error": {
                        "message": "Upstream service temporarily unavailable",
                        "type": "service_error",
                        "code": "upstream_error"
                    }
                }
            else:
                error_response = {
                    "error": {
                        "message": "Request processing failed",
                        "type": "api_error",
                        "code": "unknown_error"
                    }
                }
            
            return JSONResponse(content=error_response, status_code=e.response.status_code)
        
    else:
        async def stream_with_token_count():
            completion_tokens = 0
            completion_text = ""
            done_received = False
            upstream_usage_chunk = None  # Store upstream usage chunk if any
            
            async for chunk in stream_proxy_with_fc_transform(upstream_url, request_body_dict, headers, body.model, has_function_call, GLOBAL_TRIGGER_SIGNAL):
                # Check if this is the [DONE] marker
                if chunk.startswith(b"data: "):
                    try:
                        line_data = chunk[6:].decode('utf-8').strip()
                        if line_data == "[DONE]":
                            done_received = True
                            # Don't yield the [DONE] marker yet, we'll send it after usage info
                            break
                        elif line_data:
                            chunk_json = json.loads(line_data)
                            
                            # Check if this chunk contains usage information
                            if "usage" in chunk_json:
                                upstream_usage_chunk = chunk_json
                                logger.debug(f"🔧 Detected upstream usage chunk: {chunk_json['usage']}")
                                # Don't yield upstream usage chunk yet, we'll process it
                                continue
                            
                            # Process regular content chunks
                            if "choices" in chunk_json and len(chunk_json["choices"]) > 0:
                                delta = chunk_json["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    completion_text += content
                    except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
                        logger.debug(f"Failed to parse chunk for token counting: {e}")
                        pass
                
                yield chunk
            
            # Calculate our estimated tokens
            estimated_completion_tokens = token_counter.count_text_tokens(completion_text, body.model) if completion_text else 0
            estimated_prompt_tokens = prompt_tokens
            estimated_total_tokens = estimated_prompt_tokens + estimated_completion_tokens
            elapsed_time = time.time() - start_time
            
            # Determine final usage
            final_usage = None
            if upstream_usage_chunk and "usage" in upstream_usage_chunk:
                # Respect upstream usage, but replace zero values
                upstream_usage = upstream_usage_chunk["usage"]
                final_usage = upstream_usage.copy()
                
                if not final_usage.get("prompt_tokens") or final_usage.get("prompt_tokens") == 0:
                    final_usage["prompt_tokens"] = estimated_prompt_tokens
                    logger.debug(f"🔧 Replaced zero/missing prompt_tokens with estimate: {estimated_prompt_tokens}")
                
                if not final_usage.get("completion_tokens") or final_usage.get("completion_tokens") == 0:
                    final_usage["completion_tokens"] = estimated_completion_tokens
                    logger.debug(f"🔧 Replaced zero/missing completion_tokens with estimate: {estimated_completion_tokens}")
                
                if not final_usage.get("total_tokens") or final_usage.get("total_tokens") == 0:
                    final_usage["total_tokens"] = final_usage.get("prompt_tokens", estimated_prompt_tokens) + final_usage.get("completion_tokens", estimated_completion_tokens)
                    logger.debug(f"🔧 Replaced zero/missing total_tokens with calculated value: {final_usage['total_tokens']}")
                
                logger.debug(f"🔧 Using upstream usage with replacements: {final_usage}")
            else:
                # No upstream usage, use our estimates
                final_usage = {
                    "prompt_tokens": estimated_prompt_tokens,
                    "completion_tokens": estimated_completion_tokens,
                    "total_tokens": estimated_total_tokens
                }
                logger.debug(f"🔧 No upstream usage found, using estimates")
            
            # Log token statistics
            logger.info("=" * 60)
            logger.info(f"📊 Token Usage Statistics - Model: {body.model}")
            logger.info(f"   Input Tokens: {final_usage['prompt_tokens']}")
            logger.info(f"   Output Tokens: {final_usage['completion_tokens']}")
            logger.info(f"   Total Tokens: {final_usage['total_tokens']}")
            logger.info(f"   Duration: {elapsed_time:.2f}s")
            logger.info("=" * 60)
            
            # Send usage information if requested via stream_options OR if upstream provided usage
            if (body.stream_options and body.stream_options.get("include_usage", False)) or upstream_usage_chunk:
                usage_chunk_to_send = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": body.model,
                    "choices": [],
                    "usage": final_usage
                }
                
                # If upstream provided additional fields in the usage chunk, preserve them
                if upstream_usage_chunk:
                    for key in upstream_usage_chunk:
                        if key not in ["usage", "choices"] and key not in usage_chunk_to_send:
                            usage_chunk_to_send[key] = upstream_usage_chunk[key]
                
                yield f"data: {json.dumps(usage_chunk_to_send)}\n\n".encode('utf-8')
                logger.debug(f"🔧 Sent usage chunk in stream: {usage_chunk_to_send['usage']}")
            
            # Send [DONE] marker if it was received
            if done_received:
                yield b"data: [DONE]\n\n"
        
        return StreamingResponse(
            stream_with_token_count(),
            media_type="text/event-stream"
        )

async def stream_proxy_with_fc_transform(url: str, body: dict, headers: dict, model: str, has_fc: bool, trigger_signal: str):
    """
    Enhanced streaming proxy, supports dynamic trigger signals, avoids misjudgment within think tags
    """
    logger.info(f"📝 Starting streaming response from: {url}")
    logger.info(f"📝 Function calling enabled: {has_fc}")

    if not has_fc or not trigger_signal:
        try:
            async with http_client.stream("POST", url, json=body, headers=headers, timeout=app_config.server.timeout) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.RemoteProtocolError:
            logger.debug("🔧 Upstream closed connection prematurely, ending stream response")
            return
        return
# setattr()``
    detector = StreamingFunctionCallDetector(trigger_signal)

    def _prepare_tool_calls(parsed_tools: List[Dict[str, Any]]):
        tool_calls = []
        for i, tool in enumerate(parsed_tools):
            tool_call_id = f"call_{uuid.uuid4().hex}"
            store_tool_call_mapping(
                tool_call_id,
                tool["name"],
                tool["args"],
                f"Calling tool {tool['name']}"
            )
            tool_calls.append({
                "index": i, "id": tool_call_id, "type": "function",
                "function": { "name": tool["name"], "arguments": json.dumps(tool["args"]) }
            })
        return tool_calls

    def _build_tool_call_sse_chunks(parsed_tools: List[Dict[str, Any]], model_id: str) -> List[str]:
        tool_calls = _prepare_tool_calls(parsed_tools)
        chunks: List[str] = []

        initial_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion.chunk",
            "created": int(os.path.getmtime(__file__)), "model": model_id,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": None, "tool_calls": tool_calls}, "finish_reason": None}],
        }
        chunks.append(f"data: {json.dumps(initial_chunk)}\n\n")


        final_chunk = {
             "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion.chunk",
            "created": int(os.path.getmtime(__file__)), "model": model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        }
        chunks.append(f"data: {json.dumps(final_chunk)}\n\n")
        chunks.append("data: [DONE]\n\n")
        return chunks

    try:
        async with http_client.stream("POST", url, json=body, headers=headers, timeout=app_config.server.timeout) as response:
            if response.status_code != 200:
                error_content = await response.aread()
                logger.error(f"❌ Upstream service stream response error: status_code={response.status_code}")
                logger.error(f"❌ Upstream error details: {error_content.decode('utf-8', errors='ignore')}")
                
                if response.status_code == 401:
                    error_message = "Authentication failed"
                elif response.status_code == 403:
                    error_message = "Access forbidden"
                elif response.status_code == 429:
                    error_message = "Rate limit exceeded"
                elif response.status_code >= 500:
                    error_message = "Upstream service temporarily unavailable"
                else:
                    error_message = "Request processing failed"
                
                error_chunk = {"error": {"message": error_message, "type": "upstream_error"}}
                yield f"data: {json.dumps(error_chunk)}\n\n".encode('utf-8')
                yield b"data: [DONE]\n\n"
                return

            async for line in response.aiter_lines():
                if detector.state == "tool_parsing":
                    if line.startswith("data:"):
                        line_data = line[len("data: "):].strip()
                        if line_data and line_data != "[DONE]":
                            try:
                                chunk_json = json.loads(line_data)
                                delta_content = chunk_json.get("choices", [{}])[0].get("delta", {}).get("content", "") or ""
                                detector.content_buffer += delta_content
                                # Early termination: once </function_calls> appears, parse and finish immediately
                                if "</function_calls>" in detector.content_buffer:
                                    logger.debug("🔧 Detected </function_calls> in stream, finalizing early...")
                                    parsed_tools = detector.finalize()
                                    if parsed_tools:
                                        logger.debug(f"🔧 Early finalize: parsed {len(parsed_tools)} tool calls")
                                        for sse in _build_tool_call_sse_chunks(parsed_tools, model):
                                            yield sse.encode('utf-8')
                                        return
                                    else:
                                        logger.error("❌ Early finalize failed to parse tool calls")
                                        error_content = "Error: Detected tool use signal but failed to parse function call format"
                                        error_chunk = { "id": "error-chunk", "choices": [{"delta": {"content": error_content}}]}
                                        yield f"data: {json.dumps(error_chunk)}\n\n".encode('utf-8')
                                        yield b"data: [DONE]\n\n"
                                        return
                            except (json.JSONDecodeError, IndexError):
                                pass
                    continue
                
                if line.startswith("data:"):
                    line_data = line[len("data: "):].strip()
                    if not line_data or line_data == "[DONE]":
                        continue
                    
                    try:
                        chunk_json = json.loads(line_data)
                        delta_content = chunk_json.get("choices", [{}])[0].get("delta", {}).get("content", "") or ""
                        
                        if delta_content:
                            is_detected, content_to_yield = detector.process_chunk(delta_content)
                            
                            if content_to_yield:
                                yield_chunk = {
                                    "id": f"chatcmpl-passthrough-{uuid.uuid4().hex}",
                                    "object": "chat.completion.chunk",
                                    "created": int(os.path.getmtime(__file__)),
                                    "model": model,
                                    "choices": [{"index": 0, "delta": {"content": content_to_yield}}]
                                }
                                yield f"data: {json.dumps(yield_chunk)}\n\n".encode('utf-8')
                            
                            if is_detected:
                                # Tool call signal detected, switch to parsing mode
                                continue
                    
                    except (json.JSONDecodeError, IndexError):
                        yield line + "\n\n"

    except httpx.RequestError as e:
        logger.error(f"❌ Failed to connect to upstream service: {e}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        
        error_message = "Failed to connect to upstream service"
        error_chunk = {"error": {"message": error_message, "type": "connection_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n".encode('utf-8')
        yield b"data: [DONE]\n\n"
        return

    if detector.state == "tool_parsing":
        logger.debug(f"🔧 Stream ended, starting to parse tool call XML...")
        parsed_tools = detector.finalize()
        if parsed_tools:
            logger.debug(f"🔧 Streaming processing: Successfully parsed {len(parsed_tools)} tool calls")
            for sse in _build_tool_call_sse_chunks(parsed_tools, model):
                yield sse
            return
        else:
            logger.error(f"❌ Detected tool call signal but XML parsing failed, buffer content: {detector.content_buffer}")
            error_content = "Error: Detected tool use signal but failed to parse function call format"
            error_chunk = { "id": "error-chunk", "choices": [{"delta": {"content": error_content}}]}
            yield f"data: {json.dumps(error_chunk)}\n\n".encode('utf-8')

    elif detector.state == "detecting" and detector.content_buffer:
        # If stream has ended but buffer still has remaining characters insufficient to form signal, output them
        final_yield_chunk = {
            "id": f"chatcmpl-finalflush-{uuid.uuid4().hex}", "object": "chat.completion.chunk",
            "created": int(os.path.getmtime(__file__)), "model": model,
            "choices": [{"index": 0, "delta": {"content": detector.content_buffer}}]
        }
        yield f"data: {json.dumps(final_yield_chunk)}\n\n".encode('utf-8')

    yield b"data: [DONE]\n\n"


@app.get("/")
def read_root():
    return {
        "status": "OpenAI Function Call Middleware is running",
        "config": {
            "upstream_services_count": len(app_config.upstream_services),
            "client_keys_count": len(app_config.client_authentication.allowed_keys),
            "models_count": len(MODEL_TO_SERVICE_MAPPING),
            "features": {
                "function_calling": app_config.features.enable_function_calling,
                "log_level": app_config.features.log_level,
                "convert_developer_to_system": app_config.features.convert_developer_to_system,
                "random_trigger": True
            }
        }
    }

@app.get("/v1/models")
async def list_models(_api_key: str = Depends(verify_api_key)):
    """List all available models"""
    visible_models = set()
    for model_name in MODEL_TO_SERVICE_MAPPING.keys():
        if ':' in model_name:
            parts = model_name.split(':', 1)
            if len(parts) == 2:
                alias, _ = parts
                visible_models.add(alias)
            else:
                visible_models.add(model_name)
        else:
            visible_models.add(model_name)

    models = []
    for model_id in sorted(visible_models):
        models.append({
            "id": model_id,
            "object": "model",
            "created": 1677610602,
            "owned_by": "openai",
            "permission": [],
            "root": model_id,
            "parent": None
        })
    
    return {
        "object": "list",
        "data": models
    }


def validate_message_structure(messages: List[Dict[str, Any]]) -> bool:
    """Validate if message structure meets requirements"""
    try:
        valid_roles = ["system", "user", "assistant", "tool"]
        if not app_config.features.convert_developer_to_system:
            valid_roles.append("developer")
        
        for i, msg in enumerate(messages):
            if "role" not in msg:
                logger.error(f"❌ Message {i} missing role field")
                return False
            
            if msg["role"] not in valid_roles:
                logger.error(f"❌ Invalid role value for message {i}: {msg['role']}")
                return False
            
            if msg["role"] == "tool":
                if "tool_call_id" not in msg:
                    logger.error(f"❌ Tool message {i} missing tool_call_id field")
                    return False
            
            content = msg.get("content")
            content_info = ""
            if content:
                if isinstance(content, str):
                    content_info = f", content=text({len(content)} chars)"
                elif isinstance(content, list):
                    text_parts = [item for item in content if isinstance(item, dict) and item.get('type') == 'text']
                    image_parts = [item for item in content if isinstance(item, dict) and item.get('type') == 'image_url']
                    content_info = f", content=multimodal(text={len(text_parts)}, images={len(image_parts)})"
                else:
                    content_info = f", content={type(content).__name__}"
            else:
                content_info = ", content=empty"
            
            logger.debug(f"✅ Message {i} validation passed: role={msg['role']}{content_info}")
        
        logger.debug(f"✅ All messages validated successfully, total {len(messages)} messages")
        return True
    except Exception as e:
        logger.error(f"❌ Message validation exception: {e}")
        return False

def safe_process_tool_choice(tool_choice) -> str:
    """Safely process tool_choice field to avoid type errors"""
    try:
        if tool_choice is None:
            return ""
        
        if isinstance(tool_choice, str):
            if tool_choice == "none":
                return "\n\n**IMPORTANT:** You are prohibited from using any tools in this round. Please respond like a normal chat assistant and answer the user's question directly."
            else:
                logger.debug(f"🔧 Unknown tool_choice string value: {tool_choice}")
                return ""
        
        elif hasattr(tool_choice, 'function') and hasattr(tool_choice.function, 'name'):
            required_tool_name = tool_choice.function.name
            return f"\n\n**IMPORTANT:** In this round, you must use ONLY the tool named `{required_tool_name}`. Generate the necessary parameters and output in the specified XML format."
        
        else:
            logger.debug(f"🔧 Unsupported tool_choice type: {type(tool_choice)}")
            return ""
    
    except Exception as e:
        logger.error(f"❌ Error processing tool_choice: {e}")
        return ""

if __name__ == "__main__":
    import uvicorn
    logger.info(f"🚀 Starting server on {app_config.server.host}:{app_config.server.port}")
    logger.info(f"⏱️  Request timeout: {app_config.server.timeout} seconds")
    
    uvicorn.run(
        app,
        host=app_config.server.host,
        port=app_config.server.port,
        log_level=app_config.features.log_level.lower() if app_config.features.log_level != "DISABLED" else "critical"
    )