#!/usr/bin/env python3
"""
Test client for Outlook MCP Server
"""

import asyncio
import json
import logging
from typing import Dict, Any

from fastmcp import FastMCPClient
from fastmcp.utilities.logging import configure_logging, get_logger

# Configure logging
configure_logging(level="INFO")
logger = get_logger(__name__)

async def test_outlook_server():
    """Test the Outlook MCP server functionality."""
    
    # Create client
    client = FastMCPClient(
        name="OutlookTestClient",
        server_url="http://localhost:9027/sse"
    )
    
    try:
        # Test context with mock user_id
        test_context = {
            "metadata": {
                "user_id": "test_user_123"
            }
        }
        
        logger.info("Testing Outlook MCP Server...")
        
        # Test 1: Get folders
        logger.info("Test 1: Getting email folders...")
        try:
            folders_result = await client.call_tool(
                "get_folders",
                context=test_context
            )
            logger.info(f"Folders result: {json.dumps(folders_result, indent=2)}")
        except Exception as e:
            logger.error(f"Error getting folders: {e}")
        
        # Test 2: Get emails from inbox
        logger.info("Test 2: Getting emails from inbox...")
        try:
            emails_result = await client.call_tool(
                "get_emails",
                context=test_context,
                folder="inbox",
                top=5
            )
            logger.info(f"Emails result: {json.dumps(emails_result, indent=2)}")
        except Exception as e:
            logger.error(f"Error getting emails: {e}")
        
        # Test 3: Search emails
        logger.info("Test 3: Searching emails...")
        try:
            search_result = await client.call_tool(
                "search_emails",
                context=test_context,
                query="test",
                top=3
            )
            logger.info(f"Search result: {json.dumps(search_result, indent=2)}")
        except Exception as e:
            logger.error(f"Error searching emails: {e}")
        
        logger.info("Outlook MCP Server tests completed!")
        
    except Exception as e:
        logger.error(f"Error testing Outlook MCP server: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_outlook_server())
