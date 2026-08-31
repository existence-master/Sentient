"""
Omniroute Client for Sentient - BUYASOUL Blood Flow Integration
================================================================

Routes LLM calls through Omniroute on port 20128.
Omniroute is the blood flow - NEVER killed, NEVER duplicated.

This client replaces Sentient's LiteLLM routing with our MCP-powered model router.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Omniroute is the blood flow - always on :20128
OMNIROUTE_URL = "http://127.0.0.1:20128"
OMNIROUTE_TIMEOUT = 60

try:
    import httpx
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False
    logger.warning("httpx not available - Omniroute client will use fallback")


class OmnirouteClient:
    """
    Omniroute Client - Routes through the blood flow.
    
    Features:
    - Auto-detects if Omniroute is running
    - Falls back to configured LLM if blood flow is down
    - Routes tool calls through MCP hub
    - Maintains connection health
    """
    
    def __init__(self, fallback_url: str = None, fallback_model: str = None):
        self.base_url = OMNIROUTE_URL
        self.fallback_url = fallback_url
        self.fallback_model = fallback_model
        self.available = False
        self.last_health_check = None
        self.request_count = 0
        
    async def check_health(self) -> bool:
        """Check if Omniroute is alive."""
        if not HTTP_AVAILABLE:
            return False
            
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                response = await client.get(f"{self.base_url}/v1/models")
                if response.status_code == 200:
                    self.available = True
                    self.last_health_check = datetime.now().isoformat()
                    return True
        except Exception as e:
            logger.debug(f"Omniroute health check failed: {e}")
        
        self.available = False
        return False
    
    async def chat_completion(
        self,
        messages: List[Dict],
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: List[Dict] = None,
        **kwargs
    ) -> Dict:
        """
        Send chat completion through Omniroute.
        
        Falls back to configured LLM if Omniroute is down.
        """
        self.request_count += 1
        
        # Try Omniroute first
        if self.available or await self.check_health():
            try:
                return await self._omniroute_completion(
                    messages, model or "qwen-turbo", temperature, max_tokens, tools, **kwargs
                )
            except Exception as e:
                logger.warning(f"Omniroute request failed, falling back: {e}")
        
        # Fallback to direct LLM
        if self.fallback_url:
            return await self._fallback_completion(
                messages, model or self.fallback_model, temperature, max_tokens, **kwargs
            )
        
        raise ConnectionError("No LLM available - Omniroute down and no fallback configured")
    
    async def _omniroute_completion(
        self, messages, model, temperature, max_tokens, tools, **kwargs
    ) -> Dict:
        """Route through Omniroute."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,  # Non-streaming for integration
            **kwargs
        }
        
        async with httpx.AsyncClient(timeout=OMNIROUTE_TIMEOUT) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",  # Omniroute's actual endpoint
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise Exception(f"Omniroute error: {response.status_code} - {response.text}")
            
            result = response.json()
            result["_source"] = "omniroute"
            result["_model"] = model
            return result
    
    async def _fallback_completion(self, messages, model, temperature, max_tokens, **kwargs) -> Dict:
        """Fallback to direct LLM."""
        if not self.fallback_url:
            raise ConnectionError("No fallback URL configured")
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        async with httpx.AsyncClient(timeout=OMNIROUTE_TIMEOUT) as client:
            response = await client.post(
                f"{self.fallback_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise Exception(f"Fallback LLM error: {response.status_code}")
            
            result = response.json()
            result["_source"] = "fallback"
            result["_model"] = model
            return result
    
    async def tool_call(self, tool_name: str, arguments: Dict) -> Dict:
        """Execute a tool call through Omniroute's MCP hub."""
        if not self.available:
            if not await self.check_health():
                raise ConnectionError("Omniroute not available for tool calls")
        
        payload = {
            "name": tool_name,
            "arguments": arguments
        }
        
        async with httpx.AsyncClient(timeout=OMNIROUTE_TIMEOUT) as client:
            response = await client.post(
                f"{self.base_url}/v1/tools/call",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise Exception(f"Tool call error: {response.status_code}")
            
            return response.json()
    
    def get_status(self) -> Dict:
        """Get Omniroute client status."""
        return {
            "available": self.available,
            "url": self.base_url,
            "requests_made": self.request_count,
            "last_health_check": self.last_health_check,
            "fallback_configured": bool(self.fallback_url),
        }


# Global client instance
omniroute_client = OmnirouteClient()