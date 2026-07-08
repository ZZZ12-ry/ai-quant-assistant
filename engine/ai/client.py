"""DeepSeek API 客户端 - 使用http.client直连"""
from pathlib import Path
import yaml
import json
import http.client
import ssl
import socket
from typing import Optional


class DeepSeekClient:
    """DeepSeek API 客户端"""

    def __init__(self):
        self._config_path = Path(__file__).parent.parent.parent / "app" / "config.yaml"

    def _load_config(self) -> dict:
        with open(self._config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def chat(self, messages: list, temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> dict:
        config = self._load_config()
        api_key = config.get("api_key", "")
        model = config.get("model", "deepseek-chat")
        timeout = int(config.get("timeout", 90))
        retries = int(config.get("retries", 1))

        if not api_key:
            return {"content": "", "error": "请先配置DeepSeek API Key", "usage": None}

        data = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": temperature or config.get("temperature", 0.7),
            "max_tokens": max_tokens or config.get("max_tokens", 4096),
            "stream": False
        })

        last_error = None
        for attempt in range(retries + 1):
            conn = None
            try:
                ctx = ssl.create_default_context()
                conn = http.client.HTTPSConnection(
                    "api.deepseek.com", 443, timeout=timeout, context=ctx
                )
                conn.request(
                    "POST", "/v1/chat/completions",
                    body=data.encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                        "Host": "api.deepseek.com"
                    }
                )
                resp = conn.getresponse()
                body = resp.read().decode()

                if resp.status != 200:
                    return {"content": "", "error": f"API error {resp.status}: {body[:200]}", "usage": None}

                result = json.loads(body)
                return {
                    "content": result["choices"][0]["message"]["content"],
                    "error": None,
                    "usage": result.get("usage")
                }
            except (TimeoutError, socket.timeout, http.client.HTTPException, OSError) as e:
                last_error = e
                if attempt < retries:
                    continue
            except Exception as e:
                return {"content": "", "error": f"连接错误: {str(e)[:200]}", "usage": None}
            finally:
                try:
                    if conn:
                        conn.close()
                except Exception:
                    pass
        return {"content": "", "error": f"连接错误: {str(last_error)[:200]}。请稍后重试，或减少本次输入长度。", "usage": None}


_client = None

def get_client() -> DeepSeekClient:
    global _client
    if _client is None:
        _client = DeepSeekClient()
    return _client
