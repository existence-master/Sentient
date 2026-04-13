import os
import asyncio
import json
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from composio import Composio
from main.config import COMPOSIO_API_KEY
from fastmcp.utilities.logging import configure_logging, get_logger

from . import auth, prompts

# --- Standardized Logging Setup ---
configure_logging(level="INFO")
logger = get_logger(__name__)

# --- LLM and Environment Configuration ---
ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev-local')
if ENVIRONMENT == 'dev-local':
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)

composio = Composio(api_key=COMPOSIO_API_KEY)

mcp = FastMCP(
    name="GDocsServer",
    instructions="Provides tools to create, search, read, modify, and share Google Docs documents.",
)

@mcp.resource("prompt://gdocs-agent-system")
def get_gdocs_system_prompt() -> str:
    """Provides the system prompt that instructs the main orchestrator agent on how to use the gdocs tools."""
    return prompts.MAIN_AGENT_SYSTEM_PROMPT

async def _execute_tool(ctx: Context, action_name: str, **kwargs) -> Dict[str, Any]:
    """Helper to handle auth and execution for all tools using Composio."""
    try:
        user_id = auth.get_user_id_from_context(ctx)
        connection_id = await auth.get_composio_connection_id(user_id, "gdocs")
        
        result = await asyncio.to_thread(
            composio.tools.execute,
            action_name,
            arguments=kwargs,
            connected_account_id=connection_id,
            dangerously_skip_version_check=True
        )

        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Tool execution failed for action '{action_name}': {e}", exc_info=True)
        return {"status": "failure", "error": str(e)}

@mcp.tool()
async def copy_document(ctx: Context, document_id: str, title: Optional[str] = None) -> Dict:
    """Create a copy of an existing google document."""
    return await _execute_tool(ctx, "GOOGLEDOCS_COPY_DOCUMENT", document_id=document_id, title=title)

@mcp.tool()
async def create_document_markdown(ctx: Context, markdown_text: str, title: str) -> Dict:
    """Creates a new google docs document with markdown text."""
    return await _execute_tool(ctx, "GOOGLEDOCS_CREATE_DOCUMENT_MARKDOWN", markdown_text=markdown_text, title=title)

@mcp.tool()
async def create_document(ctx: Context, title: str, text: str = "") -> Dict:
    """Creates a new google docs document."""
    return await _execute_tool(ctx, "GOOGLEDOCS_CREATE_DOCUMENT", title=title, text=text)

@mcp.tool()
async def get_document_by_id(ctx: Context, document_id: str) -> Dict:
    """Retrieves an existing google document by its id."""
    return await _execute_tool(ctx, "GOOGLEDOCS_GET_DOCUMENT_BY_ID", id=document_id)

@mcp.tool()
async def search_documents(ctx: Context, query: Optional[str] = None, created_after: Optional[str] = None, include_trashed: Optional[bool] = None, max_results: int = 10, modified_after: Optional[str] = None, order_by: str = "modifiedTime desc", shared_with_me: Optional[bool] = None, starred_only: Optional[bool] = None) -> Dict:
    """Search for google documents using various filters."""
    return await _execute_tool(ctx, "GOOGLEDOCS_SEARCH_DOCUMENTS", query=query, created_after=created_after, include_trashed=include_trashed, max_results=max_results, modified_after=modified_after, order_by=order_by, shared_with_me=shared_with_me, starred_only=starred_only)

@mcp.tool()
async def update_document_markdown(ctx: Context, document_id: str, new_markdown_text: str) -> Dict:
    """Replaces the entire content of an existing google docs document with new markdown text."""
    return await _execute_tool(ctx, "GOOGLEDOCS_UPDATE_DOCUMENT_MARKDOWN", document_id=document_id, new_markdown_text=new_markdown_text)

@mcp.tool()
async def update_existing_document(ctx: Context, document_id: str, editDocs: List[Dict]) -> Dict:
    """Applies programmatic edits to a specified google doc."""
    return await _execute_tool(ctx, "GOOGLEDOCS_UPDATE_EXISTING_DOCUMENT", document_id=document_id, editDocs=editDocs)

@mcp.tool()
async def create_footer(ctx: Context, document_id: str, type: str, section_break_location: Optional[Dict] = None) -> Dict:
    """Create a new footer in a google document."""
    return await _execute_tool(ctx, "GOOGLEDOCS_CREATE_FOOTER", document_id=document_id, type=type, section_break_location=section_break_location)

@mcp.tool()
async def create_footnote(ctx: Context, document_id: str, endOfSegmentLocation: Optional[Dict] = None, location: Optional[Dict] = None) -> Dict:
    """Create a new footnote in a google document."""
    return await _execute_tool(ctx, "GOOGLEDOCS_CREATE_FOOTNOTE", documentId=document_id, endOfSegmentLocation=endOfSegmentLocation, location=location)

@mcp.tool()
async def create_header(ctx: Context, document_id: str, type: str, section_break_location: Optional[Dict] = None) -> Dict:
    """Create a new header in a google document."""
    createHeader_payload = {"type": type}
    if section_break_location:
        createHeader_payload["section_break_location"] = section_break_location
    return await _execute_tool(ctx, "GOOGLEDOCS_CREATE_HEADER", createHeader=createHeader_payload, documentId=document_id)

@mcp.tool()
async def create_named_range(ctx: Context, document_id: str, name: str, rangeEndIndex: int, rangeStartIndex: int, rangeSegmentId: Optional[str] = None) -> Dict:
    """Create a new named range in a google document."""
    return await _execute_tool(ctx, "GOOGLEDOCS_CREATE_NAMED_RANGE", documentId=document_id, name=name, rangeEndIndex=rangeEndIndex, rangeStartIndex=rangeStartIndex, rangeSegmentId=rangeSegmentId)

@mcp.tool()
async def create_paragraph_bullets(ctx: Context, createParagraphBullets: Dict, document_id: str) -> Dict:
    """Add bullets to paragraphs within a specified range in a google document."""
    return await _execute_tool(ctx, "GOOGLEDOCS_CREATE_PARAGRAPH_BULLETS", createParagraphBullets=createParagraphBullets, document_id=document_id)

@mcp.tool()
async def delete_content_range(ctx: Context, document_id: str, range: Dict) -> Dict:
    """Delete a range of content from a google document."""
    return await _execute_tool(ctx, "GOOGLEDOCS_DELETE_CONTENT_RANGE", document_id=document_id, range=range)

@mcp.tool()
async def delete_footer(ctx: Context, document_id: str, footer_id: str, tab_id: Optional[str] = None) -> Dict:
    """Delete a footer from a google document."""
    return await _execute_tool(ctx, "GOOGLEDOCS_DELETE_FOOTER", document_id=document_id, footer_id=footer_id, tab_id=tab_id)

@mcp.tool()
async def delete_header(ctx: Context, document_id: str, header_id: str, tab_id: Optional[str] = None) -> Dict:
    """Deletes the header from the specified section."""
    return await _execute_tool(ctx, "GOOGLEDOCS_DELETE_HEADER", document_id=document_id, header_id=header_id, tab_id=tab_id)

@mcp.tool()
async def delete_named_range(ctx: Context, deleteNamedRange: Dict, document_id: str) -> Dict:
    """Delete a named range from a google document."""
    return await _execute_tool(ctx, "GOOGLEDOCS_DELETE_NAMED_RANGE", deleteNamedRange=deleteNamedRange, document_id=document_id)

@mcp.tool()
async def delete_paragraph_bullets(ctx: Context, document_id: str, range: Dict, tab_id: Optional[str] = None) -> Dict:
    """Remove bullets from paragraphs within a specified range."""
    return await _execute_tool(ctx, "GOOGLEDOCS_DELETE_PARAGRAPH_BULLETS", document_id=document_id, range=range, tab_id=tab_id)

@mcp.tool()
async def delete_table(ctx: Context, document_id: str, table_end_index: int, table_start_index: int, segment_id: Optional[str] = None, tab_id: Optional[str] = None) -> Dict:
    """Delete an entire table from a google document."""
    return await _execute_tool(ctx, "GOOGLEDOCS_DELETE_TABLE", document_id=document_id, table_end_index=table_end_index, table_start_index=table_start_index, segment_id=segment_id, tab_id=tab_id)

@mcp.tool()
async def delete_table_column(ctx: Context, document_id: str, requests: List[Dict]) -> Dict:
    """Delete a column from a table in a google document."""
    return await _execute_tool(ctx, "GOOGLEDOCS_DELETE_TABLE_COLUMN", document_id=document_id, requests=requests)

@mcp.tool()
async def delete_table_row(ctx: Context, document_id: str, tableCellLocation: Dict) -> Dict:
    """Delete a row from a table in a google document."""
    return await _execute_tool(ctx, "GOOGLEDOCS_DELETE_TABLE_ROW", documentId=document_id, tableCellLocation=tableCellLocation)

@mcp.tool()
async def get_charts_from_spreadsheet(ctx: Context, spreadsheet_id: str) -> Dict:
    """Retrieve a list of all charts from a specified google sheets spreadsheet."""
    return await _execute_tool(ctx, "GOOGLEDOCS_GET_CHARTS_FROM_SPREADSHEET", spreadsheet_id=spreadsheet_id)

@mcp.tool()
async def insert_inline_image(ctx: Context, document_id: str, location: Dict, uri: str, objectSize: Optional[Dict] = None) -> Dict:
    """Insert an image from a given uri at a specified location."""
    return await _execute_tool(ctx, "GOOGLEDOCS_INSERT_INLINE_IMAGE", documentId=document_id, location=location, uri=uri, objectSize=objectSize)

@mcp.tool()
async def insert_page_break(ctx: Context, document_id: str, insertPageBreak: Dict) -> Dict:
    """Insert a page break into a google document."""
    return await _execute_tool(ctx, "GOOGLEDOCS_INSERT_PAGE_BREAK", documentId=document_id, insertPageBreak=insertPageBreak)

@mcp.tool()
async def insert_table_action(ctx: Context, columns: int, document_id: str, rows: int, index: Optional[int] = None, insertAtEndOfSegment: Optional[bool] = None, segmentId: Optional[str] = None, tabId: Optional[str] = None) -> Dict:
    """Insert a table into a google document."""
    return await _execute_tool(ctx, "GOOGLEDOCS_INSERT_TABLE_ACTION", columns=columns, documentId=document_id, rows=rows, index=index, insertAtEndOfSegment=insertAtEndOfSegment, segmentId=segmentId, tabId=tabId)

@mcp.tool()
async def insert_table_column(ctx: Context, document_id: str, requests: List[Dict]) -> Dict:
    """Insert a new column into a table in a google document."""
    return await _execute_tool(ctx, "GOOGLEDOCS_INSERT_TABLE_COLUMN", document_id=document_id, requests=requests)

@mcp.tool()
async def insert_text_action(ctx: Context, document_id: str, insertion_index: int, text_to_insert: str) -> Dict:
    """Insert a string of text at a specified location within a google document."""
    return await _execute_tool(ctx, "GOOGLEDOCS_INSERT_TEXT_ACTION", document_id=document_id, insertion_index=insertion_index, text_to_insert=text_to_insert)

@mcp.tool()
async def list_spreadsheet_charts_action(ctx: Context, spreadsheet_id: str, fields_mask: str = "sheets(properties(sheetId,title),charts(chartId,spec(title,altText)))") -> Dict:
    """Retrieve a list of charts with their ids and metadata from a google sheets spreadsheet."""
    return await _execute_tool(ctx, "GOOGLEDOCS_LIST_SPREADSHEET_CHARTS_ACTION", spreadsheet_id=spreadsheet_id, fields_mask=fields_mask)

@mcp.tool()
async def replace_all_text(ctx: Context, document_id: str, find_text: str, match_case: bool, replace_text: str, search_by_regex: Optional[bool] = None, tab_ids: Optional[List[str]] = None) -> Dict:
    """Replace all occurrences of a specified text string with another text string."""
    return await _execute_tool(ctx, "GOOGLEDOCS_REPLACE_ALL_TEXT", document_id=document_id, find_text=find_text, match_case=match_case, replace_text=replace_text, search_by_regex=search_by_regex, tab_ids=tab_ids)

@mcp.tool()
async def replace_image(ctx: Context, document_id: str, replace_image: Dict) -> Dict:
    """Replace a specific image in a document with a new image from a uri."""
    return await _execute_tool(ctx, "GOOGLEDOCS_REPLACE_IMAGE", document_id=document_id, replace_image=replace_image)

@mcp.tool()
async def unmerge_table_cells(ctx: Context, document_id: str, tableRange: Dict) -> Dict:
    """Unmerge previously merged cells in a table."""
    return await _execute_tool(ctx, "GOOGLEDOCS_UNMERGE_TABLE_CELLS", document_id=document_id, tableRange=tableRange)

@mcp.tool()
async def update_document_style(ctx: Context, document_id: str, document_style: Dict, fields: str, tab_id: Optional[str] = None) -> Dict:
    """Update the overall document style."""
    return await _execute_tool(ctx, "GOOGLEDOCS_UPDATE_DOCUMENT_STYLE", document_id=document_id, document_style=document_style, fields=fields, tab_id=tab_id)

@mcp.tool()
async def update_table_row_style(ctx: Context, document_id: str, updateTableRowStyle: Dict) -> Dict:
    """Update the style of a table row in a google document."""
    return await _execute_tool(ctx, "GOOGLEDOCS_UPDATE_TABLE_ROW_STYLE", documentId=document_id, updateTableRowStyle=updateTableRowStyle)

if __name__ == "__main__":
    host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_SERVER_PORT", 9004))
    
    print(f"Starting GDocs MCP Server on http://{host}:{port}")
    mcp.run(transport="sse", host=host, port=port)