def test_dsv4_encoding_roundtrip():
    from inference_lib.encoding import encode_messages, parse_message_from_completion_text

    # chat mode without tools — simple prompt encode
    msgs = [{"role": "user", "content": "hello"}]
    prompt = encode_messages(msgs, thinking_mode="chat")
    assert isinstance(prompt, str) and len(prompt) > 0
    assert "hello" in prompt
    # thinking mode parse example — use actual encoded prompt slice
    msgs2 = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "capital of France?"},
        {
            "role": "assistant",
            "reasoning_content": "The user asks about the capital of France. It is Paris.",
            "content": "The capital of France is Paris.",
        },
    ]
    prompt2 = encode_messages(msgs2, thinking_mode="thinking")
    marker = "<｜Assistant｜><think>"
    last_start = prompt2.rfind(marker) + len(marker)
    parsed = parse_message_from_completion_text(prompt2[last_start:], thinking_mode="thinking")
    assert parsed["reasoning_content"] == "The user asks about the capital of France. It is Paris."
    assert "Paris" in parsed["content"]
