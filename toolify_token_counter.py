# SPDX-License-Identifier: GPL-3.0-or-later
#
# Toolify: Empower any LLM with function calling capabilities and token counting.
# Copyright (C) 2025 FunnyCups (https://github.com/funnycups)

"""
统一服务: 
1. 函数调用增强代理服务 - 给任何LLM添加函数调用功能
2. Token计数代理服务 - 精确计算和监控token使用情况
"""

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
import uvicorn
from typing import List, Dict, Any, Optional, Literal, Union, AsyncGenerator
from collections import OrderedDict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Header, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError, validator, field_validator
from openai import AsyncOpenAI

from config_loader import config_loader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==================== Token 计算器 ====================
class TokenCounter:
    """使用 tiktoken 计算 token 数量"""
    
    def __init__(self):
        self.encoders = {}
    
    def get_encoder(self, model: str):
        """获取或创建对应模型的编码器"""
        if model not in self.encoders:
            try:
                # 尝试获取模型专用编码器
                self.encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # 默认使用
                logger.warning(f"Model {model} not found, using o200k_base encoding")
                self.encoders[model] = tiktoken.get_encoding("o200k_base")
        return self.encoders[model]
    
    def count_tokens(self, messages: list, model: str = "gpt-5-high") -> int:
        """计算消息列表的 token 数量"""
        encoder = self.get_encoder(model)
        
        # 根据模型选择计算方式
        if model.startswith("gpt-5-high") or model.startswith("gpt-4"):
            return self._count_chat_tokens(messages, encoder, model)
        else:
            # 简单计算：拼接所有文本内容
            text_content = []
            for msg in messages:
                content = msg.get("content", "")
                # 处理content可能是列表的情况(多模态消息)
                if isinstance(content, list):
                    # 只提取列表中的文本部分
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_content.append(item.get("text", ""))
                elif isinstance(content, str):
                    text_content.append(content)
            
            # 拼接所有文本
            full_text = " ".join(text_content)
            return len(encoder.encode(full_text))
    
    def _count_chat_tokens(self, messages: list, encoder, model: str) -> int:
        """Chat 模型的精确 token 计算"""
        tokens_per_message = 3  # 每条消息的固定开销
        tokens_per_name = 1      # 如果有 name 字段
        
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if key == "content":
                    # 处理content可能是列表的情况(多模态消息)
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict) and item.get("type") == "text":
                                content_text = item.get("text", "")
                                num_tokens += len(encoder.encode(content_text))
                    elif isinstance(value, str):
                        num_tokens += len(encoder.encode(value))
                elif key == "name":
                    num_tokens += tokens_per_name
                    if isinstance(value, str):
                        num_tokens += len(encoder.encode(value))
                elif isinstance(value, str):
                    num_tokens += len(encoder.encode(value))
        
        num_tokens += 3  # 每次回复都有固定的 priming
        return num_tokens
    
    def count_text_tokens(self, text: str, model: str = "gpt-5-high") -> int:
        """计算纯文本的 token 数量"""
        encoder = self.get_encoder(model)
        return len(encoder.encode(text))

# 全局 token 计数器
token_counter = TokenCounter()

# ==================== 工具调用映射管理器 ====================
class ToolCallMappingManager:
    """
    工具调用映射管理器，具有TTL（生存时间）和大小限制
    
    功能：
    1. 自动过期清理 - 条目在指定时间后自动删除
    2. 大小限制 - 防止无限内存增长
    3. LRU淘汰 - 在达到大小限制时删除最近最少使用的条目
    4. 线程安全 - 支持并发访问
    5. 定期清理 - 后台线程定期清理过期条目
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600, cleanup_interval: int = 300):
        """
        初始化映射管理器
        
        参数：
            max_size: 存储条目的最大数量
            ttl_seconds: 条目生存时间（秒）
            cleanup_interval: 清理间隔（秒）
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval
        
        self._data: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()
        
        logger.debug(f"🔧 [INIT] 工具调用映射管理器已启动 - 最大条目: {max_size}, TTL: {ttl_seconds}秒, 清理间隔: {cleanup_interval}秒")
    
    def store(self, tool_call_id: str, name: str, args: dict, description: str = "") -> None:
        """存储工具调用映射"""
        with self._lock:
            current_time = time.time()
            
            if tool_call_id in self._data:
                del self._data[tool_call_id]
                del self._timestamps[tool_call_id]
            
            while len(self._data) >= self.max_size:
                oldest_key = next(iter(self._data))
                del self._data[oldest_key]
                del self._timestamps[oldest_key]
                logger.debug(f"🔧 [CLEANUP] 因大小限制删除了最旧条目: {oldest_key}")
            
            self._data[tool_call_id] = {
                "name": name,
                "args": args,
                "description": description,
                "created_at": current_time
            }
            self._timestamps[tool_call_id] = current_time
            
            logger.debug(f"🔧 存储了工具调用映射: {tool_call_id} -> {name}")
            logger.debug(f"🔧 当前映射表大小: {len(self._data)}")
    
    def get(self, tool_call_id: str) -> Optional[Dict[str, Any]]:
        """获取工具调用映射（更新LRU顺序）"""
        with self._lock:
            current_time = time.time()
            
            if tool_call_id not in self._data:
                logger.debug(f"🔧 未找到工具调用映射: {tool_call_id}")
                logger.debug(f"🔧 当前映射表中的所有ID: {list(self._data.keys())}")
                return None
            
            if current_time - self._timestamps[tool_call_id] > self.ttl_seconds:
                logger.debug(f"🔧 工具调用映射已过期: {tool_call_id}")
                del self._data[tool_call_id]
                del self._timestamps[tool_call_id]
                return None
            
            result = self._data[tool_call_id]
            self._data.move_to_end(tool_call_id)
            
            logger.debug(f"🔧 找到工具调用映射: {tool_call_id} -> {result['name']}")
            return result
    
    def cleanup_expired(self) -> int:
        """清理过期条目，返回清理的条目数量"""
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
                logger.debug(f"🔧 [CLEANUP] 已清理 {len(expired_keys)} 个过期条目")
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
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
        """后台定期清理线程"""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                cleaned = self.cleanup_expired()
                
                stats = self.get_stats()
                if stats["total_entries"] > 0:
                    logger.debug(f"🔧 [STATS] 映射表状态: 总计={stats['total_entries']}, "
                               f"活动={stats['active_entries']}, 内存使用率={stats['memory_usage_ratio']:.1%}")
                
            except Exception as e:
                logger.error(f"❌ 后台清理线程异常: {e}")

# ==================== 辅助函数 ====================
def generate_random_trigger_signal() -> str:
    """生成一个随机的、自闭合的触发信号，如 <Function_AB1c_Start/>"""
    chars = string.ascii_letters + string.digits
    random_str = ''.join(secrets.choice(chars) for _ in range(4))
    return f"<Function_{random_str}_Start/>"

# ==================== 配置加载 ====================
try:
    # 加载配置文件
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
    
    logger.info(f"✅ 配置加载成功: {config_loader.config_path}")
    logger.info(f"📊 已配置 {len(app_config.upstream_services)} 个上游服务")
    logger.info(f"🔑 已配置 {len(app_config.client_authentication.allowed_keys)} 个客户端密钥")
    
    MODEL_TO_SERVICE_MAPPING, ALIAS_MAPPING = config_loader.get_model_to_service_mapping()
    DEFAULT_SERVICE = config_loader.get_default_service()
    ALLOWED_CLIENT_KEYS = config_loader.get_allowed_client_keys()
    GLOBAL_TRIGGER_SIGNAL = generate_random_trigger_signal()
    
    logger.info(f"🎯 已配置 {len(MODEL_TO_SERVICE_MAPPING)} 个模型映射")
    if ALIAS_MAPPING:
        logger.info(f"🔄 已配置 {len(ALIAS_MAPPING)} 个模型别名: {list(ALIAS_MAPPING.keys())}")
    logger.info(f"🔄 默认服务: {DEFAULT_SERVICE['name']}")
    
except Exception as e:
    logger.error(f"❌ 配置加载失败: {type(e).__name__}")
    logger.error(f"❌ 错误详情: {str(e)}")
    logger.error("💡 请确保config.yaml文件存在并且格式正确")
    
    # 使用环境变量配置Token计数器代理
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-123456")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:5108/v1")
    PROXY_PORT = int(os.getenv("PROXY_PORT", "5112"))
    PROXY_HOST = os.getenv("PROXY_HOST", "0.0.0.0")
    logger.info(f"✅ 使用环境变量配置Token计数器代理")
    logger.info(f"✅ OpenAI基础URL: {OPENAI_BASE_URL}")

# ==================== 映射函数 ====================
def store_tool_call_mapping(tool_call_id: str, name: str, args: dict, description: str = ""):
    """存储工具调用ID与调用内容之间的映射"""
    TOOL_CALL_MAPPING_MANAGER.store(tool_call_id, name, args, description)

def get_tool_call_mapping(tool_call_id: str) -> Optional[Dict[str, Any]]:
    """获取对应于工具调用ID的调用内容"""
    return TOOL_CALL_MAPPING_MANAGER.get(tool_call_id)

def format_tool_result_for_ai(tool_call_id: str, result_content: str) -> str:
    """格式化工具调用结果，使AI能够理解，使用英文提示和XML结构"""
    logger.debug(f"🔧 格式化工具调用结果: tool_call_id={tool_call_id}")
    tool_info = get_tool_call_mapping(tool_call_id)
    if not tool_info:
        logger.debug(f"🔧 未找到工具调用映射，使用默认格式")
        return f"Tool execution result:\n<tool_result>\n{result_content}\n</tool_result>"
    
    formatted_text = f"""Tool execution result:
- Tool name: {tool_info['name']}
- Execution result:
<tool_result>
{result_content}
</tool_result>"""
    
    logger.debug(f"🔧 格式化完成，工具名称: {tool_info['name']}")
    return formatted_text

def format_assistant_tool_calls_for_ai(tool_calls: List[Dict[str, Any]], trigger_signal: str) -> str:
    """将助手工具调用格式化为AI可读的字符串格式"""
    logger.debug(f"🔧 格式化助手工具调用. 数量: {len(tool_calls)}")
    
    xml_calls_parts = []
    for tool_call in tool_calls:
        function_info = tool_call.get("function", {})
        name = function_info.get("name", "")
        arguments_json = function_info.get("arguments", "{}")
        
        try:
            # 首先尝试作为JSON加载。如果是有效的JSON字符串，则解析它。
            args_dict = json.loads(arguments_json)
        except (json.JSONDecodeError, TypeError):
            # 如果不是有效的JSON字符串，则将其视为简单字符串。
            args_dict = {"raw_arguments": arguments_json}

        args_parts = []
        for key, value in args_dict.items():
            # 将值转换回JSON字符串，以便在XML内部一致表示。
            json_value = json.dumps(value, ensure_ascii=False)
            args_parts.append(f"<{key}>{json_value}</{key}>")
        
        args_content = "\n".join(args_parts)
        
        xml_call = f"<function_call>\n<tool>{name}</tool>\n<args>\n{args_content}\n</args>\n</function_call>"
        xml_calls_parts.append(xml_call)

    all_calls = "\n".join(xml_calls_parts)
    final_str = f"{trigger_signal}\n<function_calls>\n{all_calls}\n</function_calls>"
    
    logger.debug("🔧 助手工具调用格式化成功。")
    return final_str

def get_function_call_prompt_template(trigger_signal: str) -> str:
    """
    基于动态触发信号生成提示模板
    """
    custom_template = app_config.features.prompt_template
    if custom_template:
        logger.info("🔧 使用配置中的自定义提示模板")
        return custom_template.format(
            trigger_signal=trigger_signal,
            tools_list="{tools_list}"
        )
    
    return f"""
您可以访问以下可用工具来帮助解决问题:

{{tools_list}}

**重要上下文说明:**
1. 您可以在单个响应中调用多个工具。
2. 对话上下文中可能已经包含了来自先前函数调用的工具执行结果。仔细查看对话历史，以避免不必要的重复工具调用。
3. 当上下文中存在工具执行结果时，它们将使用<tool_result>...</tool_result>等XML标签进行格式化，以便于识别。
4. 这是唯一可以用于工具调用的格式，任何偏差都将导致失败。

当您需要使用工具时，必须严格遵循此格式。请勿在工具调用语法的第一行和第二行添加任何额外的文本、解释或对话:

1. 开始工具调用时，在新行上准确输入:
{trigger_signal}
前后没有空格，完全按照上面所示输出。触发信号必须独自占一行，且只出现一次。

2. 从第二行开始，立即跟上完整的<function_calls> XML块。

3. 对于多个工具调用，在同一个<function_calls>包装器中包含多个<function_call>块。

4. 不要在结束的</function_calls>标签后添加任何文本或解释。

严格的参数键规则:
- 您必须完全按照定义使用参数键（区分大小写和标点符号）。不要重命名、添加或删除字符。
- 如果键以连字符开头（例如-i、-C），您必须在标签名称中保留连字符。例如：<-i>true</-i>，<-C>2</-C>。
- 切勿将"-i"转换为"i"或将"-C"转换为"C"。不要将参数键复数化、翻译或别名。
- <tool>标签必须包含列表中工具的确切名称。任何其他工具名称都是无效的。
- <args>必须包含该工具的所有必需参数。

正确示例（多个工具调用，包括连字符键）:
...响应内容（可选）...
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

错误示例（额外文本+错误的键名称 - 不要这样做）:
...响应内容（可选）...
{trigger_signal}
我将为您调用工具。
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

现在请准备严格按照上述规范操作。
"""

def remove_think_blocks(text: str) -> str:
    """
    暂时删除所有<think>...</think>块以便XML解析
    支持嵌套think标签
    注意：此函数仅用于临时解析，不影响返回给用户的原始内容
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

def parse_function_calls_xml(xml_string: str, trigger_signal: str) -> Optional[List[Dict[str, Any]]]:
    """
    增强的XML解析函数，支持动态触发信号
    1. 保留<think>...</think>块（应该正常返回给用户）
    2. 仅在解析function_calls时暂时删除think块，以防止think内容干扰XML解析
    3. 查找触发信号的最后一次出现
    4. 从最后一个触发信号开始解析function_calls
    """
    logger.debug(f"🔧 改进的解析器开始处理，输入长度: {len(xml_string) if xml_string else 0}")
    logger.debug(f"🔧 使用触发信号: {trigger_signal[:20]}...")
    
    if not xml_string or trigger_signal not in xml_string:
        logger.debug(f"🔧 输入为空或不包含触发信号")
        return None
    
    cleaned_content = remove_think_blocks(xml_string)
    logger.debug(f"🔧 临时删除think块后的内容长度: {len(cleaned_content)}")
    
    signal_positions = []
    start_pos = 0
    while True:
        pos = cleaned_content.find(trigger_signal, start_pos)
        if pos == -1:
            break
        signal_positions.append(pos)
        start_pos = pos + 1
    
    if not signal_positions:
        logger.debug(f"🔧 在清理后的内容中未找到触发信号")
        return None
    
    logger.debug(f"🔧 找到 {len(signal_positions)} 个触发信号位置: {signal_positions}")
    
    last_signal_pos = signal_positions[-1]
    content_after_signal = cleaned_content[last_signal_pos:]
    logger.debug(f"🔧 从最后一个触发信号开始的内容: {repr(content_after_signal[:100])}")
    
    calls_content_match = re.search(r"<function_calls>([\s\S]*?)</function_calls>", content_after_signal)
    if not calls_content_match:
        logger.debug(f"🔧 未找到function_calls标签")
        return None
    
    calls_content = calls_content_match.group(1)
    logger.debug(f"🔧 function_calls内容: {repr(calls_content)}")
    
    results = []
    call_blocks = re.findall(r"<function_call>([\s\S]*?)</function_call>", calls_content)
    logger.debug(f"🔧 找到 {len(call_blocks)} 个function_call块")
    
    for i, block in enumerate(call_blocks):
        logger.debug(f"🔧 处理function_call #{i+1}: {repr(block)}")
        
        tool_match = re.search(r"<tool>(.*?)</tool>", block)
        if not tool_match:
            logger.debug(f"🔧 在块 #{i+1} 中未找到tool标签")
            continue
        
        name = tool_match.group(1).strip()
        args = {}
        
        args_block_match = re.search(r"<args>([\s\S]*?)</args>", block)
        if args_block_match:
            args_content = args_block_match.group(1)
            # 支持包含连字符的参数标签名称（如-i、-A）；匹配任何非空格、非'>'和非'/'字符
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
        logger.debug(f"🔧 添加了工具调用: {result}")
    
    logger.debug(f"🔧 最终解析结果: {results}")
    return results if results else None

class StreamingFunctionCallDetector:
    """增强的流式函数调用检测器，支持动态触发信号，避免在<think>标签内误判
    
    核心功能：
    1. 避免在<think>块内触发工具调用检测
    2. 正常将<think>块内容输出给用户
    3. 支持嵌套think标签
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
        处理流式内容块
        返回: (是否检测到工具调用, 要输出的内容)
        """
        if not delta_content:
            return False, ""
        
        self.content_buffer += delta_content
        content_to_yield = ""
        
        if self.state == "tool_parsing":
            return False, ""
        
        if delta_content:
            logger.debug(f"🔧 处理块: {repr(delta_content[:50])}{'...' if len(delta_content) > 50 else ''}, 缓冲区长度: {len(self.content_buffer)}, think状态: {self.in_think_block}")
        
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
                    logger.debug(f"🔧 改进的检测器: 在非think块中检测到触发信号! 信号: {self.signal[:20]}...")
                    logger.debug(f"🔧 触发信号位置: {i}, think状态: {self.in_think_block}, think深度: {self.think_depth}")
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
        """更新think标签状态，支持嵌套"""
        remaining = self.content_buffer[pos:]
        
        if remaining.startswith('<think>'):
            self.think_depth += 1
            self.in_think_block = True
            logger.debug(f"🔧 进入think块，深度: {self.think_depth}")
            return 7
        
        elif remaining.startswith('</think>'):
            self.think_depth = max(0, self.think_depth - 1)
            self.in_think_block = self.think_depth > 0
            logger.debug(f"🔧 退出think块，深度: {self.think_depth}")
            return 8
        
        return 0
    
    def _can_detect_signal_at(self, pos: int) -> bool:
        """检查是否可以在指定位置检测信号"""
        return (pos + self.signal_len <= len(self.content_buffer) and 
                not self.in_think_block)
    
    def finalize(self) -> Optional[List[Dict[str, Any]]]:
        """流结束时的最终处理"""
        if self.state == "tool_parsing":
            return parse_function_calls_xml(self.content_buffer, self.trigger_signal)
        return None

# ==================== 请求/响应模型 ====================
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
    
    model_config = {"extra": "allow"}

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
    
    model_config = {"extra": "allow"}
    
    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        """验证消息格式"""
        if not v:
            raise ValueError("Messages list cannot be empty")
        for msg in v:
            if not isinstance(msg, dict):
                raise ValueError("Each message must be a dictionary")
            if 'role' not in msg:
                raise ValueError("Each message must contain 'role' field")
        return v

def generate_function_prompt(tools: List[Tool], trigger_signal: str) -> tuple[str, str]:
    """
    根据客户端请求中的工具定义生成注入的系统提示。
    返回: (提示内容, 触发信号)
    """
    tools_list_str = []
    for i, tool in enumerate(tools):
        func = tool.function
        name = func.name
        description = func.description or ""

        # 强健地读取JSON Schema字段
        schema: Dict[str, Any] = func.parameters or {}
        props: Dict[str, Any] = schema.get("properties", {}) or {}
        required_list: List[str] = schema.get("required", []) or []

        # 简短摘要行: name (type)
        params_summary = ", ".join([
            f"{p_name} ({(p_info or {}).get('type', 'any')})" for p_name, p_info in props.items()
        ]) or "None"

        # 为提示注入构建详细参数规范（默认启用）
        detail_lines: List[str] = []
        for p_name, p_info in props.items():
            p_info = p_info or {}
            p_type = p_info.get("type", "any")
            is_required = "Yes" if p_name in required_list else "No"
            p_desc = p_info.get("description")
            enum_vals = p_info.get("enum")
            default_val = p_info.get("default")
            examples_val = p_info.get("examples") or p_info.get("example")

            # 常见约束和提示
            constraints: Dict[str, Any] = {}
            for key in [
                "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                "minLength", "maxLength", "pattern", "format",
                "minItems", "maxItems", "uniqueItems"
            ]:
                if key in p_info:
                    constraints[key] = p_info.get(key)

            # 数组项类型提示
            if p_type == "array":
                items = p_info.get("items") or {}
                if isinstance(items, dict):
                    itype = items.get("type")
                    if itype:
                        constraints["items.type"] = itype

            # 组成漂亮的行
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

def find_upstream(model_name: str) -> tuple[Dict[str, Any], str]:
    """根据模型名称查找上游配置，处理别名和直通模式。"""
    
    # 处理模型直通模式
    if app_config.features.model_passthrough:
        logger.info("🔄 模型直通模式处于活动状态。转发到'openai'服务。")
        openai_service = None
        for service in app_config.upstream_services:
            if service.name == "openai":
                openai_service = service.model_dump()
                break
        
        if openai_service:
            if not openai_service.get("api_key"):
                 raise HTTPException(status_code=500, detail="配置错误: 在模型直通模式下未找到'openai'服务的API密钥。")
            # 在直通模式下，直接使用请求中的模型名称。
            return openai_service, model_name
        else:
            raise HTTPException(status_code=500, detail="配置错误: 启用了'model_passthrough'，但未找到名为'openai'的上游服务。")

    # 默认路由逻辑
    chosen_model_entry = model_name
    
    if model_name in ALIAS_MAPPING:
        chosen_model_entry = random.choice(ALIAS_MAPPING[model_name])
        logger.info(f"🔄 检测到模型别名'{model_name}'。为此请求随机选择了'{chosen_model_entry}'。")

    service = MODEL_TO_SERVICE_MAPPING.get(chosen_model_entry)
    
    if service:
        if not service.get("api_key"):
            raise HTTPException(status_code=500, detail=f"模型配置错误: 未找到服务'{service.get('name')}'的API密钥。")
    else:
        logger.warning(f"⚠️  配置中未找到模型'{model_name}'，使用默认服务")
        service = DEFAULT_SERVICE
        if not service.get("api_key"):
            raise HTTPException(status_code=500, detail="服务配置错误: 未找到默认API密钥。")

    actual_model_name = chosen_model_entry
    if ':' in chosen_model_entry:
         parts = chosen_model_entry.split(':', 1)
         if len(parts) == 2:
             _, actual_model_name = parts
            
    return service, actual_model_name

def validate_message_structure(messages: List[Dict[str, Any]]) -> bool:
    """验证消息结构是否符合要求"""
    try:
        valid_roles = ["system", "user", "assistant", "tool"]
        if not app_config.features.convert_developer_to_system:
            valid_roles.append("developer")
        
        for i, msg in enumerate(messages):
            if "role" not in msg:
                logger.error(f"❌ 消息 {i} 缺少role字段")
                return False
            
            if msg["role"] not in valid_roles:
                logger.error(f"❌ 消息 {i} 的role值无效: {msg['role']}")
                return False
            
            if msg["role"] == "tool":
                if "tool_call_id" not in msg:
                    logger.error(f"❌ 工具消息 {i} 缺少tool_call_id字段")
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
            
            logger.debug(f"✅ 消息 {i} 验证通过: role={msg['role']}{content_info}")
        
        logger.debug(f"✅ 所有消息验证成功，总共 {len(messages)} 条消息")
        return True
    except Exception as e:
        logger.error(f"❌ 消息验证异常: {e}")
        return False

def safe_process_tool_choice(tool_choice) -> str:
    """安全处理tool_choice字段，避免类型错误"""
    try:
        if tool_choice is None:
            return ""
        
        if isinstance(tool_choice, str):
            if tool_choice == "none":
                return "\n\n**重要:** 您被禁止在本轮对话中使用任何工具。请像正常的聊天助手一样回应并直接回答用户的问题。"
            else:
                logger.debug(f"🔧 未知的tool_choice字符串值: {tool_choice}")
                return ""
        
        elif hasattr(tool_choice, 'function') and hasattr(tool_choice.function, 'name'):
            required_tool_name = tool_choice.function.name
            return f"\n\n**重要:** 在本轮对话中，您必须仅使用名为`{required_tool_name}`的工具。生成必要的参数并以指定的XML格式输出。"
        
        else:
            logger.debug(f"🔧 不支持的tool_choice类型: {type(tool_choice)}")
            return ""
    
    except Exception as e:
        logger.error(f"❌ 处理tool_choice时出错: {e}")
        return ""

def preprocess_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """预处理消息，将工具类型消息转换为AI可理解的格式，返回字典列表以避免Pydantic验证问题"""
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
                    logger.debug(f"🔧 将工具消息转换为用户消息: tool_call_id={tool_call_id}")
                else:
                    logger.debug(f"🔧 跳过无效的工具消息: tool_call_id={tool_call_id}, content={bool(content)}")
            elif message.get("role") == "assistant" and "tool_calls" in message and message["tool_calls"]:
                tool_calls = message.get("tool_calls", [])
                formatted_tool_calls_str = format_assistant_tool_calls_for_ai(tool_calls, GLOBAL_TRIGGER_SIGNAL)
                
                # 与原始内容组合（如果存在）
                original_content = message.get("content") or ""
                final_content = f"{original_content}\n{formatted_tool_calls_str}".strip()

                processed_message = {
                    "role": "assistant",
                    "content": final_content
                }
                # 从原始消息复制其他潜在键，除了tool_calls
                for key, value in message.items():
                    if key not in ["role", "content", "tool_calls"]:
                        processed_message[key] = value

                processed_messages.append(processed_message)
                logger.debug(f"🔧 将assistant tool_calls转换为content。")

            elif message.get("role") == "developer":
                if app_config.features.convert_developer_to_system:
                    processed_message = message.copy()
                    processed_message["role"] = "system"
                    processed_messages.append(processed_message)
                    logger.debug(f"🔧 将developer消息转换为system消息，以便更好地与上游兼容")
                else:
                    processed_messages.append(message)
                    logger.debug(f"🔧 保持developer角色不变（基于配置）")
            else:
                processed_messages.append(message)
        else:
            processed_messages.append(message)
    
    return processed_messages

# ==================== OpenAI 客户端和 FastAPI 应用 ====================
TOOL_CALL_MAPPING_MANAGER = ToolCallMappingManager(
    max_size=1000,
    ttl_seconds=3600,
    cleanup_interval=300
)

openai_client = None
http_client = httpx.AsyncClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global openai_client
    # 启动时初始化
    if 'OPENAI_API_KEY' in globals() and 'OPENAI_BASE_URL' in globals():
        openai_client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        logger.info(f"OpenAI 客户端初始化完成，base_url: {OPENAI_BASE_URL}")
    
    yield
    
    # 关闭时清理
    if openai_client:
        await openai_client.close()
    await http_client.aclose()
    logger.info("HTTP和OpenAI客户端已关闭")

app = FastAPI(
    title="Toolify Token Counter",
    description="集成了函数调用增强和Token计数功能的代理服务",
    version="1.0.0",
    lifespan=lifespan
)

# ==================== 中间件和异常处理 ====================
@app.middleware("http")
async def debug_middleware(request: Request, call_next):
    """用于调试验证错误的中间件，不记录对话内容。"""
    response = await call_next(request)
    
    if response.status_code == 422:
        logger.debug(f"🔍 检测到{request.method} {request.url.path}的验证错误")
        logger.debug(f"🔍 响应状态码: 422 (Pydantic验证失败)")
    
    return response

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """处理Pydantic验证错误，提供详细的错误信息"""
    logger.error(f"❌ Pydantic验证错误: {exc}")
    logger.error(f"❌ 请求URL: {request.url}")
    logger.error(f"❌ 错误详情: {exc.errors()}")
    
    for error in exc.errors():
        logger.error(f"❌ 验证错误位置: {error.get('loc')}")
        logger.error(f"❌ 验证错误消息: {error.get('msg')}")
        logger.error(f"❌ 验证错误类型: {error.get('type')}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "请求格式无效",
                "type": "invalid_request_error",
                "code": "invalid_request"
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """处理所有未捕获的异常"""
    logger.error(f"❌ 未处理的异常: {exc}")
    logger.error(f"❌ 请求URL: {request.url}")
    logger.error(f"❌ 异常类型: {type(exc).__name__}")
    logger.error(f"❌ 错误堆栈: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "内部服务器错误",
                "type": "server_error",
                "code": "internal_error"
            }
        }
    )

async def verify_api_key(authorization: str = Header(...)):
    """依赖项: 验证客户端API密钥"""
    client_key = authorization.replace("Bearer ", "")
    if app_config.features.key_passthrough:
        # 在直通模式下，跳过allowed_keys检查
        return client_key
    if client_key not in ALLOWED_CLIENT_KEYS:
        raise HTTPException(status_code=401, detail="未授权")
    return client_key

# ==================== 流式响应处理 ====================
async def stream_response(response, model: str, prompt_tokens: int, start_time: float, request_id: str = None) -> AsyncGenerator[str, None]:
    """处理流式响应（针对token计数）- 支持OpenAI对象和字符串行两种输入"""
    completion_tokens = 0
    completion_text = ""
    last_chunk_data = None
    
    try:
        async for chunk in response:
            # 处理字符串类型的输入 (来自httpx.aiter_lines())
            if isinstance(chunk, str):
                # 只处理SSE格式的数据行
                if chunk.startswith("data: "):
                    line_data = chunk[len("data: "):].strip()
                    
                    # 处理结束标记
                    if line_data == "[DONE]":
                        continue
                    
                    # 尝试解析JSON
                    if line_data:
                        try:
                            chunk_json = json.loads(line_data)
                            
                            # 从JSON中提取内容
                            if "choices" in chunk_json and len(chunk_json["choices"]) > 0:
                                delta = chunk_json["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    completion_text += content
                            
                            # 保存最后一个chunk数据
                            last_chunk_data = chunk_json
                            
                            # 直接传递原始SSE格式数据
                            yield chunk + "\n\n"
                            
                            # 检查是否有usage信息
                            if "usage" in chunk_json:
                                completion_tokens = chunk_json["usage"].get("completion_tokens", 0)
                        except json.JSONDecodeError:
                            # 无法解析为JSON的行，原样传递
                            yield chunk + "\n\n"
                else:
                    # 非SSE格式数据，原样传递
                    yield chunk + "\n\n"
            
            # 处理对象类型的输入 (来自OpenAI API)
            elif hasattr(chunk, 'choices') and chunk.choices:
                # 收集完成文本以计算token
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    completion_text += delta.content
                
                # 保存最后一个chunk的数据（用于构建usage chunk）
                last_chunk_data = chunk
                
                # 转换为 OpenAI 格式的 SSE，但不发送usage信息
                chunk_dict = chunk.model_dump()
                # 移除可能存在的usage字段，我们会在最后添加
                if 'usage' in chunk_dict:
                    chunk_dict.pop('usage')
                yield f"data: {json.dumps(chunk_dict)}\n\n"
                
                # 检查是否有usage信息（OpenAI在最后一个chunk中提供）
                if hasattr(chunk, 'usage') and chunk.usage:
                    completion_tokens = chunk.usage.completion_tokens
        
        # 如果没有从API获取到completion_tokens，则手动计算
        if completion_tokens == 0 and completion_text:
            completion_tokens = token_counter.count_text_tokens(completion_text, model)
        
        total_tokens = prompt_tokens + completion_tokens
        elapsed_time = time.time() - start_time
        
        # 输出token统计信息
        logger.info("=" * 60)
        logger.info(f"📊 Token 使用统计 - 模型: {model}")
        logger.info(f"   输入 Tokens: {prompt_tokens}")
        logger.info(f"   输出 Tokens: {completion_tokens}")
        logger.info(f"   总计 Tokens: {total_tokens}")
        logger.info(f"   耗时: {elapsed_time:.2f}秒")
        logger.info("=" * 60)
        
        # 发送包含代理计算的usage信息的chunk（符合OpenAI格式）
        if last_chunk_data:
            usage_chunk = {
                "id": last_chunk_data.id if hasattr(last_chunk_data, 'id') else f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": last_chunk_data.created if hasattr(last_chunk_data, 'created') else int(time.time()),
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"
        
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"流错误: {e}")
        yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

async def stream_proxy_with_fc_transform(url: str, body: dict, headers: dict, model: str, has_fc: bool, trigger_signal: str):
    """
    增强型流式代理，支持动态触发信号，避免在think标签内误判
    """
    logger.info(f"📝 开始来自 {url} 的流式响应")
    logger.info(f"📝 函数调用已启用: {has_fc}")

    if not has_fc or not trigger_signal:
        try:
            async with http_client.stream("POST", url, json=body, headers=headers, timeout=app_config.server.timeout) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.RemoteProtocolError:
            logger.debug("🔧 上游过早关闭连接，结束流式响应")
            return
        return

    detector = StreamingFunctionCallDetector(trigger_signal)

    def _prepare_tool_calls(parsed_tools: List[Dict[str, Any]]):
        tool_calls = []
        for i, tool in enumerate(parsed_tools):
            tool_call_id = f"call_{uuid.uuid4().hex}"
            store_tool_call_mapping(
                tool_call_id,
                tool["name"],
                tool["args"],
                f"调用工具 {tool['name']}"
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
                logger.error(f"❌ 上游服务流响应错误: status_code={response.status_code}")
                logger.error(f"❌ 上游错误详情: {error_content.decode('utf-8', errors='ignore')}")
                
                if response.status_code == 401:
                    error_message = "身份验证失败"
                elif response.status_code == 403:
                    error_message = "拒绝访问"
                elif response.status_code == 429:
                    error_message = "超出速率限制"
                elif response.status_code >= 500:
                    error_message = "上游服务暂时不可用"
                else:
                    error_message = "请求处理失败"
                
                error_chunk = {"error": {"message": error_message, "type": "upstream_error"}}
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
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
                                # 提前终止：一旦出现</function_calls>，立即解析并完成
                                if "</function_calls>" in detector.content_buffer:
                                    logger.debug("🔧 在流中检测到</function_calls>，提前结束...")
                                    parsed_tools = detector.finalize()
                                    if parsed_tools:
                                        logger.debug(f"🔧 提前完成: 解析了 {len(parsed_tools)} 个工具调用")
                                        for sse in _build_tool_call_sse_chunks(parsed_tools, model):
                                            yield sse
                                        return
                                    else:
                                        logger.error("❌ 提前完成解析工具调用失败")
                                        error_content = "错误: 检测到工具使用信号但未能解析函数调用格式"
                                        error_chunk = { "id": "error-chunk", "choices": [{"delta": {"content": error_content}}]}
                                        yield f"data: {json.dumps(error_chunk)}\n\n"
                                        yield "data: [DONE]\n\n"
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
                                yield f"data: {json.dumps(yield_chunk)}\n\n"
                            
                            if is_detected:
                                # 检测到工具调用信号，切换到解析模式
                                continue
                    
                    except (json.JSONDecodeError, IndexError):
                        yield line + "\n\n"

    except httpx.RequestError as e:
        logger.error(f"❌ 连接到上游服务失败: {e}")
        logger.error(f"❌ 错误类型: {type(e).__name__}")
        
        error_message = "连接到上游服务失败"
        error_chunk = {"error": {"message": error_message, "type": "connection_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return

    if detector.state == "tool_parsing":
        logger.debug(f"🔧 流结束，开始解析工具调用XML...")
        parsed_tools = detector.finalize()
        if parsed_tools:
            logger.debug(f"🔧 流处理：成功解析了 {len(parsed_tools)} 个工具调用")
            for sse in _build_tool_call_sse_chunks(parsed_tools, model):
                yield sse
            return
        else:
            logger.error(f"❌ 检测到工具调用信号但XML解析失败，缓冲区内容: {detector.content_buffer}")
            error_content = "错误: 检测到工具使用信号但未能解析函数调用格式"
            error_chunk = { "id": "error-chunk", "choices": [{"delta": {"content": error_content}}]}
            yield f"data: {json.dumps(error_chunk)}\n\n"

    elif detector.state == "detecting" and detector.content_buffer:
        # 如果流已结束但缓冲区仍有剩余字符不足以形成信号，输出它们
        final_yield_chunk = {
            "id": f"chatcmpl-finalflush-{uuid.uuid4().hex}", "object": "chat.completion.chunk",
            "created": int(os.path.getmtime(__file__)), "model": model,
            "choices": [{"index": 0, "delta": {"content": detector.content_buffer}}]
        }
        yield f"data: {json.dumps(final_yield_chunk)}\n\n"

    yield "data: [DONE]\n\n"

# ==================== API 端点 ====================
@app.get("/")
def read_root():
    """健康检查"""
    features_info = {}
    try:
        features_info = {
            "function_calling": app_config.features.enable_function_calling,
            "log_level": app_config.features.log_level,
            "convert_developer_to_system": app_config.features.convert_developer_to_system,
            "random_trigger": True,
            "token_counting": True
        }
    except:
        features_info = {
            "token_counting": True
        }
    
    return {
        "status": "running",
        "service": "Toolify Token Counter",
        "version": "1.0.0",
        "features": features_info
    }

@app.get("/v1/models")
async def list_models(_api_key: str = Depends(verify_api_key)):
    """列出可用模型"""
    try:
        if openai_client:
            # 如果是作为token计数代理使用，从OpenAI获取模型
            models = await openai_client.models.list()
            return models.model_dump()
        else:
            # 如果是作为函数调用代理使用，使用配置的模型
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
    except Exception as e:
        logger.error(f"列出模型时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/usage/tokens")
async def get_token_count(text: str, model: str = "gpt-5-high"):
    """获取文本的 token 数量（调试接口）"""
    try:
        tokens = token_counter.count_text_tokens(text, model)
        return {
            "text": text,
            "model": model,
            "tokens": tokens
        }
    except Exception as e:
        logger.error(f"计算token时出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _api_key: str = Depends(verify_api_key)
):
    """聊天补全接口，支持函数调用和token计数"""
    start_time = time.time()
    
    # 计算输入token
    prompt_tokens = token_counter.count_tokens(request.messages, request.model)
    logger.info(f"请求到 {request.model} - 输入tokens: {prompt_tokens}")
    
    try:
        # 检查是否使用函数调用代理
        if 'MODEL_TO_SERVICE_MAPPING' in globals() and 'GLOBAL_TRIGGER_SIGNAL' in globals():
            logger.debug(f"🔧 收到请求，模型: {request.model}")
            logger.debug(f"🔧 消息数量: {len(request.messages)}")
            logger.debug(f"🔧 工具数量: {len(request.tools) if request.tools else 0}")
            logger.debug(f"🔧 流式: {request.stream}")
            
            upstream, actual_model = find_upstream(request.model)
            upstream_url = f"{upstream['base_url']}/chat/completions"
            
            logger.debug(f"🔧 开始消息预处理，原始消息数量: {len(request.messages)}")
            processed_messages = preprocess_messages(request.messages)
            logger.debug(f"🔧 预处理完成，处理后消息数量: {len(processed_messages)}")
            
            if not validate_message_structure(processed_messages):
                logger.error(f"❌ 消息结构验证失败，但继续处理")
            
            request_body_dict = request.model_dump(exclude_unset=True)
            request_body_dict["model"] = actual_model
            request_body_dict["messages"] = processed_messages
            is_fc_enabled = app_config.features.enable_function_calling
            has_tools_in_request = bool(request.tools)
            has_function_call = is_fc_enabled and has_tools_in_request
            
            logger.debug(f"🔧 请求体构建完成，消息数量: {len(processed_messages)}")
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {_api_key}" if app_config.features.key_passthrough else f"Bearer {upstream['api_key']}",
                "Accept": "application/json" if not request.stream else "text/event-stream"
            }

            logger.info(f"📝 转发请求到上游: {upstream['name']}")
            logger.info(f"📝 模型: {request_body_dict.get('model', 'unknown')}, 消息: {len(request_body_dict.get('messages', []))}")

            if has_function_call:
                logger.debug(f"🔧 使用此请求的全局触发信号: {GLOBAL_TRIGGER_SIGNAL}")
                
                function_prompt, _ = generate_function_prompt(request.tools, GLOBAL_TRIGGER_SIGNAL)
                
                tool_choice_prompt = safe_process_tool_choice(request.tool_choice)
                if tool_choice_prompt:
                    function_prompt += tool_choice_prompt

                system_message = {"role": "system", "content": function_prompt}
                request_body_dict["messages"].insert(0, system_message)
                
                if "tools" in request_body_dict:
                    del request_body_dict["tools"]
                if "tool_choice" in request_body_dict:
                    del request_body_dict["tool_choice"]

            elif has_tools_in_request and not is_fc_enabled:
                logger.info(f"🔧 配置已禁用函数调用，忽略请求中的'tools'和'tool_choice'。")
                if "tools" in request_body_dict:
                    del request_body_dict["tools"]
                if "tool_choice" in request_body_dict:
                    del request_body_dict["tool_choice"]

            if not request.stream:
                try:
                    logger.debug(f"🔧 发送上游请求到: {upstream_url}")
                    logger.debug(f"🔧 has_function_call: {has_function_call}")
                    logger.debug(f"🔧 请求体包含tools: {bool(request.tools)}")
                    
                    upstream_response = await http_client.post(
                        upstream_url, json=request_body_dict, headers=headers, timeout=app_config.server.timeout
                    )
                    upstream_response.raise_for_status() # 如果状态码为4xx或5xx，则引发异常
                    
                    response_json = upstream_response.json()
                    logger.debug(f"🔧 上游响应状态码: {upstream_response.status_code}")
                    
                    # 计算输出token并添加token统计
                    completion_text = ""
                    if response_json.get("choices") and len(response_json["choices"]) > 0:
                        content = response_json["choices"][0].get("message", {}).get("content")
                        if content:
                            completion_text = content
                    
                    completion_tokens = token_counter.count_text_tokens(completion_text, request.model) if completion_text else 0
                    total_tokens = prompt_tokens + completion_tokens
                    elapsed_time = time.time() - start_time
                    
                    # 输出token统计信息
                    logger.info("=" * 60)
                    logger.info(f"📊 Token 使用统计 - 模型: {request.model}")
                    logger.info(f"   输入 Tokens: {prompt_tokens}")
                    logger.info(f"   输出 Tokens: {completion_tokens}")
                    logger.info(f"   总计 Tokens: {total_tokens}")
                    logger.info(f"   耗时: {elapsed_time:.2f}秒")
                    logger.info("=" * 60)
                    
                    # 添加token统计到响应
                    response_json["usage"] = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    }
                    
                    if has_function_call:
                        content = response_json["choices"][0]["message"]["content"]
                        logger.debug(f"🔧 完整响应内容: {repr(content)}")
                        
                        parsed_tools = parse_function_calls_xml(content, GLOBAL_TRIGGER_SIGNAL)
                        logger.debug(f"🔧 XML解析结果: {parsed_tools}")
                        
                        if parsed_tools:
                            logger.debug(f"🔧 成功解析 {len(parsed_tools)} 个工具调用")
                            tool_calls = []
                            for tool in parsed_tools:
                                tool_call_id = f"call_{uuid.uuid4().hex}"
                                store_tool_call_mapping(
                                    tool_call_id,
                                    tool["name"],
                                    tool["args"],
                                    f"调用工具 {tool['name']}"
                                )
                                tool_calls.append({
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool["name"],
                                        "arguments": json.dumps(tool["args"])
                                    }
                                })
                            logger.debug(f"🔧 转换后的tool_calls: {tool_calls}")
                            
                            response_json["choices"][0]["message"] = {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": tool_calls,
                            }
                            response_json["choices"][0]["finish_reason"] = "tool_calls"
                            logger.debug(f"🔧 函数调用转换完成")
                        else:
                            logger.debug(f"🔧 未检测到工具调用，返回原始内容（包括think块）")
                    else:
                        logger.debug(f"🔧 未检测到函数调用或转换条件不满足")
                    
                    return JSONResponse(content=response_json)

                except httpx.HTTPStatusError as e:
                    logger.error(f"❌ 上游服务响应错误: status_code={e.response.status_code}")
                    logger.error(f"❌ 上游错误详情: {e.response.text}")
                    
                    if e.response.status_code == 400:
                        error_response = {
                            "error": {
                                "message": "请求参数无效",
                                "type": "invalid_request_error",
                                "code": "bad_request"
                            }
                        }
                    elif e.response.status_code == 401:
                        error_response = {
                            "error": {
                                "message": "身份验证失败",
                                "type": "authentication_error", 
                                "code": "unauthorized"
                            }
                        }
                    elif e.response.status_code == 403:
                        error_response = {
                            "error": {
                                "message": "拒绝访问",
                                "type": "permission_error",
                                "code": "forbidden"
                            }
                        }
                    elif e.response.status_code == 429:
                        error_response = {
                            "error": {
                                "message": "超出速率限制",
                                "type": "rate_limit_error",
                                "code": "rate_limit_exceeded"
                            }
                        }
                    elif e.response.status_code >= 500:
                        error_response = {
                            "error": {
                                "message": "上游服务暂时不可用",
                                "type": "service_error",
                                "code": "upstream_error"
                            }
                        }
                    else:
                        error_response = {
                            "error": {
                                "message": "请求处理失败",
                                "type": "api_error",
                                "code": "unknown_error"
                            }
                        }
                    
                    return JSONResponse(content=error_response, status_code=e.response.status_code)
                
            else:
                if has_function_call:
                    return StreamingResponse(
                        stream_proxy_with_fc_transform(upstream_url, request_body_dict, headers, request.model, has_function_call, GLOBAL_TRIGGER_SIGNAL),
                        media_type="text/event-stream"
                    )
                else:
                    # 使用token计数流式处理
                    response = await http_client.post(
                        upstream_url, json=request_body_dict, headers=headers, timeout=app_config.server.timeout
                    )
                    response.raise_for_status()
                    
                    # 处理流式响应 - 直接传递字符串行
                    return StreamingResponse(
                        stream_response(response.aiter_lines(), request.model, prompt_tokens, start_time),
                        media_type="text/event-stream"
                    )
        
        # 如果不是函数调用代理，则作为token计数代理使用
        elif openai_client:
            # 直接使用 messages，已经是字典列表格式
            messages = request.messages
            
            # 记录首条消息内容类型，便于调试
            if messages and len(messages) > 0:
                first_msg = messages[0]
                content = first_msg.get("content", "")
                content_type = type(content).__name__
                logger.debug(f"First message content type: {content_type}")
                if isinstance(content, list):
                    logger.debug(f"First message content items: {len(content)}")
            
            # 调用 OpenAI API
            response = await openai_client.chat.completions.create(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                top_p=request.top_p,
                n=request.n,
                stream=request.stream,
                stop=request.stop,
                max_tokens=request.max_tokens,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                user=request.user
            )
            
            # 流式响应
            if request.stream:
                logger.info(f"开始流式响应 - 模型: {request.model}, 输入Tokens: {prompt_tokens}")
                return StreamingResponse(
                    stream_response(response, request.model, prompt_tokens, start_time),
                    media_type="text/event-stream"
                )
            
            # 非流式响应
            # 计算输出token（如果响应中有内容的话）
            completion_text = ""
            if response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    completion_text = choice.message.content or ""
            
            # 使用代理计算的token数
            completion_tokens = token_counter.count_text_tokens(completion_text, request.model) if completion_text else 0
            total_tokens = prompt_tokens + completion_tokens
            
            elapsed_time = time.time() - start_time
            
            # 输出token统计信息
            logger.info("=" * 60)
            logger.info(f"📊 Token 使用统计 - 模型: {request.model}")
            logger.info(f"   输入 Tokens: {prompt_tokens}")
            logger.info(f"   输出 Tokens: {completion_tokens}")
            logger.info(f"   总计 Tokens: {total_tokens}")
            logger.info(f"   耗时: {elapsed_time:.2f}秒")
            logger.info("=" * 60)
            
            # 返回响应，使用代理计算的usage
            response_dict = response.model_dump()
            response_dict['usage'] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
            return response_dict
            
    except Exception as e:
        logger.error(f"chat_completions中的错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def completions(request: Request):
    """文本补全接口（仅Token计数功能）"""
    try:
        body = await request.json()
        prompt = body.get("prompt", "")
        model = body.get("model", "gpt-5-high-instruct")
        
        # 计算输入 token
        start_time = time.time()
        prompt_tokens = token_counter.count_text_tokens(prompt, model)
        
        # 调用 OpenAI API
        response = await openai_client.completions.create(**body)
        
        # 计算输出token
        completion_text = ""
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, 'text'):
                completion_text = choice.text or ""
        
        # 使用代理计算的token数
        completion_tokens = token_counter.count_text_tokens(completion_text, model) if completion_text else 0
        total_tokens = prompt_tokens + completion_tokens
        elapsed_time = time.time() - start_time
        
        # 输出token统计信息
        logger.info("=" * 60)
        logger.info(f"📊 Token 使用统计 - 模型: {model}")
        logger.info(f"   输入 Tokens: {prompt_tokens}")
        logger.info(f"   输出 Tokens: {completion_tokens}")
        logger.info(f"   总计 Tokens: {total_tokens}")
        logger.info(f"   耗时: {elapsed_time:.2f}秒")
        logger.info("=" * 60)
        
        # 返回响应，使用代理计算的usage
        response_dict = response.model_dump()
        response_dict['usage'] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
        return response_dict
        
    except Exception as e:
        logger.error(f"Error in completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== 启动服务 ====================
if __name__ == "__main__":
    port = PROXY_PORT if 'PROXY_PORT' in globals() else 5115  # 使用自定义端口5115
    host = PROXY_HOST if 'PROXY_HOST' in globals() else app_config.server.host
    log_level = "info" if 'PROXY_PORT' in globals() else app_config.features.log_level.lower()
    
    if log_level == "disabled":
        log_level = "critical"
    
    logger.info(f"🚀 在 {host}:{port} 启动服务")
    
    if 'app_config' in globals():
        logger.info(f"⏱️ 请求超时: {app_config.server.timeout} 秒")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level
    )