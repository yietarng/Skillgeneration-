from trace_collection.redact import redact_text, redact_value


def test_redacts_anthropic_key():
    text = "export ANTHROPIC_API_KEY=sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"
    assert "sk-ant-" not in redact_text(text)
    assert "[REDACTED]" in redact_text(text)


def test_redacts_aws_key():
    text = "aws_access_key_id = AKIAABCDEFGHIJKLMNOP"
    assert "AKIAABCDEFGHIJKLMNOP" not in redact_text(text)


def test_redacts_private_key_block():
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIBogIBAAJ...\n-----END RSA PRIVATE KEY-----"
    result = redact_text(text)
    assert "MIIBogIBAAJ" not in result
    assert "[REDACTED]" in result


def test_leaves_benign_text_untouched():
    text = "Ran `pytest tests/` -- 12 passed, 0 failed."
    assert redact_text(text) == text


def test_redact_value_recurses_through_nested_structures():
    value = {
        "steps": [
            {"output": "token: sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"},
            {"output": "no secret here"},
        ],
        "count": 2,
    }
    redacted = redact_value(value)
    assert "sk-ant-" not in redacted["steps"][0]["output"]
    assert redacted["steps"][1]["output"] == "no secret here"
    assert redacted["count"] == 2
