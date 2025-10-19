#!/usr/bin/env python3
"""
Test script to verify token counting accuracy for tool calls
"""

import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import TokenCounter

def test_token_counting():
    """Test token counting with various message types including tool calls"""

    token_counter = TokenCounter()

    # Test 1: Simple message without tool calls
    print("=" * 60)
    print("Test 1: Simple message without tool calls")
    messages1 = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"}
    ]
    tokens1 = token_counter.count_tokens(messages1, "gpt-4")
    print(f"Messages: {json.dumps(messages1, indent=2)}")
    print(f"Token count: {tokens1}")
    print()

    # Test 2: Message with tool calls
    print("=" * 60)
    print("Test 2: Message with tool calls")
    messages2 = [
        {"role": "user", "content": "What's the weather like in New York?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "New York", "units": "celsius"})
                    }
                }
            ]
        }
    ]
    tokens2 = token_counter.count_tokens(messages2, "gpt-4")
    print(f"Messages: {json.dumps(messages2, indent=2)}")
    print(f"Token count: {tokens2}")
    print()

    # Test 3: Message with tool response
    print("=" * 60)
    print("Test 3: Message with tool response")
    messages3 = [
        {"role": "user", "content": "What's the weather like in New York?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "New York", "units": "celsius"})
                    }
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": "Temperature: 22°C, Condition: Partly cloudy"
        }
    ]
    tokens3 = token_counter.count_tokens(messages3, "gpt-4")
    print(f"Messages: {json.dumps(messages3, indent=2)}")
    print(f"Token count: {tokens3}")
    print()

    # Test 4: Multiple tool calls
    print("=" * 60)
    print("Test 4: Multiple tool calls")
    messages4 = [
        {"role": "user", "content": "Compare the weather in New York and London"},
        {
            "role": "assistant",
            "content": "I'll check the weather in both cities for you.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "New York", "units": "celsius"})
                    }
                },
                {
                    "id": "call_456",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "London", "units": "celsius"})
                    }
                }
            ]
        }
    ]
    tokens4 = token_counter.count_tokens(messages4, "gpt-4")
    print(f"Messages: {json.dumps(messages4, indent=2)}")
    print(f"Token count: {tokens4}")
    print()

    # Compare token counts
    print("=" * 60)
    print("Summary:")
    print(f"Test 1 (no tools): {tokens1} tokens")
    print(f"Test 2 (1 tool call): {tokens2} tokens")
    print(f"Test 3 (tool + response): {tokens3} tokens")
    print(f"Test 4 (2 tool calls + content): {tokens4} tokens")
    print()
    print("Token difference analysis:")
    print(f"Tool call overhead (Test 2 - Test 1): {tokens2 - tokens1} tokens")
    print(f"Tool response overhead (Test 3 - Test 2): {tokens3 - tokens2} tokens")
    print(f"Multiple tools overhead (Test 4 vs Test 2): {tokens4 - tokens2} tokens")

    # Test with different models
    print()
    print("=" * 60)
    print("Model comparison for Test 2:")
    for model in ["gpt-4", "gpt-3.5-turbo", "o1", "o3"]:
        try:
            tokens = token_counter.count_tokens(messages2, model)
            print(f"  {model}: {tokens} tokens")
        except Exception as e:
            print(f"  {model}: Error - {e}")

if __name__ == "__main__":
    test_token_counting()
