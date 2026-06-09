import httpx
from typing import Any
from dataclasses import dataclass
from ..utils.config import settings # I need to make sure utils/config exists

# For now, I'll just assume a simple way to get keys from env
import os
from dotenv import load_dotenv

load_dotenv()

class BaseConnector:
    def __init__(self):
        self.client = httpx.Client(timeout=10.0)
        self.headers = {"Authorization": f"Bearer {os.getenv('SPORTMONK_API_KEY', '')}"}

    def request(self, method: str, url: str, **kwargs) -> Any:
        response = self.client.request(method, url, headers=self.headers, **kwargs)
        response.raise_for_status()
        return response.json()

    def close(self):
        self.client.close()
