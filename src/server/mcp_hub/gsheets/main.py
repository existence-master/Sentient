import os
import asyncio
import json
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from json_extractor import JsonExtractor
from fastmcp.utilities.logging import configure_logging, get_logger
from composio import Composio
from main.config import COMPOSIO_API_KEY

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
    name="GSheetsServer",
    instructions="Provides tools to create, manage, and interact with Google Sheets, including reading/writing data and managing the spreadsheet structure.",
)

@mcp.resource("prompt://gsheets-agent-system")
def get_gsheets_system_prompt() -> str:
    return prompts.MAIN_AGENT_SYSTEM_PROMPT

async def _execute_tool(ctx: Context, action_name: str, **kwargs) -> Dict[str, Any]:
    """Helper to handle auth and execution for all tools using Composio."""
    try:
        user_id = auth.get_user_id_from_context(ctx)
        connection_id = await auth.get_composio_connection_id(user_id, "gsheets")
        
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
async def add_sheet(ctx: Context, spreadsheet_id: str, include_spreadsheet_in_response: Optional[bool] = None, properties: Optional[Dict] = None, response_include_grid_data: Optional[bool] = None) -> Dict:
    """Adds a new sheet (worksheet) to a spreadsheet."""
    return await _execute_tool(ctx, "GOOGLESHEETS_ADD_SHEET", spreadsheetId=spreadsheet_id, includeSpreadsheetInResponse=include_spreadsheet_in_response, properties=properties, responseIncludeGridData=response_include_grid_data)

@mcp.tool()
async def append_dimension(ctx: Context, dimension: str, length: int, sheet_id: int, spreadsheet_id: str) -> Dict:
    """Append new rows or columns to a sheet."""
    return await _execute_tool(ctx, "GOOGLESHEETS_APPEND_DIMENSION", dimension=dimension, length=length, sheet_id=sheet_id, spreadsheet_id=spreadsheet_id)

@mcp.tool()
async def batch_update(ctx: Context, sheet_name: str, spreadsheet_id: str, values: List[List[Any]], first_cell_location: Optional[str] = None, include_values_in_response: Optional[bool] = None, value_input_option: str = "USER_ENTERED") -> Dict:
    """Updates a specified range in a google sheet with given values."""
    return await _execute_tool(ctx, "GOOGLESHEETS_BATCH_UPDATE", sheet_name=sheet_name, spreadsheet_id=spreadsheet_id, values=values, first_cell_location=first_cell_location, includeValuesInResponse=include_values_in_response, valueInputOption=value_input_option)

@mcp.tool()
async def clear_basic_filter(ctx: Context, sheet_id: int, spreadsheet_id: str) -> Dict:
    """Clear the basic filter from a sheet."""
    return await _execute_tool(ctx, "GOOGLESHEETS_CLEAR_BASIC_FILTER", sheet_id=sheet_id, spreadsheet_id=spreadsheet_id)

@mcp.tool()
async def delete_dimension(ctx: Context, delete_dimension_request: Dict, spreadsheet_id: str, include_spreadsheet_in_response: Optional[bool] = None, response_include_grid_data: Optional[bool] = None, response_ranges: Optional[List[str]] = None) -> Dict:
    """Delete specified rows or columns from a sheet."""
    return await _execute_tool(ctx, "GOOGLESHEETS_DELETE_DIMENSION", delete_dimension_request=delete_dimension_request, spreadsheet_id=spreadsheet_id, include_spreadsheet_in_response=include_spreadsheet_in_response, response_include_grid_data=response_include_grid_data, response_ranges=response_ranges)

@mcp.tool()
async def delete_sheet(ctx: Context, sheet_id: int, spreadsheet_id: str) -> Dict:
    """Delete a sheet (worksheet) from a spreadsheet."""
    return await _execute_tool(ctx, "GOOGLESHEETS_DELETE_SHEET", sheet_id=sheet_id, spreadsheetId=spreadsheet_id)

@mcp.tool()
async def get_spreadsheet_by_data_filter(ctx: Context, data_filters: List[Dict], spreadsheet_id: str, exclude_tables_in_banded_ranges: Optional[bool] = None, include_grid_data: Optional[bool] = None) -> Dict:
    """Returns the spreadsheet at the given id, filtered by the specified data filters."""
    return await _execute_tool(ctx, "GOOGLESHEETS_GET_SPREADSHEET_BY_DATA_FILTER", dataFilters=data_filters, spreadsheetId=spreadsheet_id, excludeTablesInBandedRanges=exclude_tables_in_banded_ranges, includeGridData=include_grid_data)

@mcp.tool()
async def get_spreadsheet_info(ctx: Context, spreadsheet_id: str) -> Dict:
    """Retrieves comprehensive metadata for a google spreadsheet."""
    return await _execute_tool(ctx, "GOOGLESHEETS_GET_SPREADSHEET_INFO", spreadsheet_id=spreadsheet_id)

@mcp.tool()
async def insert_dimension(ctx: Context, insert_dimension: Dict, spreadsheet_id: str, include_spreadsheet_in_response: Optional[bool] = None, response_include_grid_data: Optional[bool] = None, response_ranges: Optional[List[str]] = None) -> Dict:
    """Insert new rows or columns into a sheet at a specified location."""
    return await _execute_tool(ctx, "GOOGLESHEETS_INSERT_DIMENSION", insert_dimension=insert_dimension, spreadsheet_id=spreadsheet_id, include_spreadsheet_in_response=include_spreadsheet_in_response, response_include_grid_data=response_include_grid_data, response_ranges=response_ranges)

@mcp.tool()
async def search_developer_metadata(ctx: Context, data_filters: List[Dict], spreadsheet_id: str) -> Dict:
    """Search for developer metadata in a spreadsheet."""
    return await _execute_tool(ctx, "GOOGLESHEETS_SEARCH_DEVELOPER_METADATA", dataFilters=data_filters, spreadsheetId=spreadsheet_id)

@mcp.tool()
async def set_basic_filter(ctx: Context, filter: Dict, spreadsheet_id: str) -> Dict:
    """Set a basic filter on a sheet in a google spreadsheet."""
    return await _execute_tool(ctx, "GOOGLESHEETS_SET_BASIC_FILTER", filter=filter, spreadsheetId=spreadsheet_id)

@mcp.tool()
async def copy_sheet_to_another_spreadsheet(ctx: Context, destination_spreadsheet_id: str, sheet_id: int, spreadsheet_id: str) -> Dict:
    """Copy a single sheet from a spreadsheet to another spreadsheet."""
    return await _execute_tool(ctx, "GOOGLESHEETS_SPREADSHEETS_SHEETS_COPY_TO", destination_spreadsheet_id=destination_spreadsheet_id, sheet_id=sheet_id, spreadsheet_id=spreadsheet_id)

@mcp.tool()
async def append_values_to_spreadsheet(ctx: Context, range: str, spreadsheet_id: str, value_input_option: str, values: List[List[Any]], include_values_in_response: Optional[bool] = None, insert_data_option: Optional[str] = None, major_dimension: Optional[str] = None, response_date_time_render_option: Optional[str] = None, response_value_render_option: Optional[str] = None) -> Dict:
    """Append values to a spreadsheet."""
    return await _execute_tool(ctx, "GOOGLESHEETS_SPREADSHEETS_VALUES_APPEND", range=range, spreadsheetId=spreadsheet_id, valueInputOption=value_input_option, values=values, includeValuesInResponse=include_values_in_response, insertDataOption=insert_data_option, majorDimension=major_dimension, responseDateTimeRenderOption=response_date_time_render_option, responseValueRenderOption=response_value_render_option)

@mcp.tool()
async def batch_clear_spreadsheet_values(ctx: Context, ranges: List[str], spreadsheet_id: str) -> Dict:
    """Clear one or more ranges of values from a spreadsheet."""
    return await _execute_tool(ctx, "GOOGLESHEETS_SPREADSHEETS_VALUES_BATCH_CLEAR", ranges=ranges, spreadsheet_id=spreadsheet_id)

@mcp.tool()
async def batch_clear_values_by_data_filter(ctx: Context, data_filters: List[Dict], spreadsheet_id: str) -> Dict:
    """Clears one or more ranges of values from a spreadsheet using data filters."""
    return await _execute_tool(ctx, "GOOGLESHEETS_SPREADSHEETS_VALUES_BATCH_CLEAR_BY_DATA_FILTER", dataFilters=data_filters, spreadsheetId=spreadsheet_id)

@mcp.tool()
async def batch_get_spreadsheet_values_by_data_filter(ctx: Context, data_filters: List[Dict], spreadsheet_id: str, date_time_render_option: Optional[str] = None, major_dimension: Optional[str] = None, value_render_option: Optional[str] = None) -> Dict:
    """Return one or more ranges of values from a spreadsheet that match the specified data filters."""
    return await _execute_tool(ctx, "GOOGLESHEETS_SPREADSHEETS_VALUES_BATCH_GET_BY_DATA_FILTER", dataFilters=data_filters, spreadsheetId=spreadsheet_id, dateTimeRenderOption=date_time_render_option, majorDimension=major_dimension, valueRenderOption=value_render_option)

@mcp.tool()
async def update_sheet_properties(ctx: Context, spreadsheet_id: str, update_sheet_properties: Dict) -> Dict:
    """Update properties of a sheet (worksheet) within a google spreadsheet."""
    return await _execute_tool(ctx, "GOOGLESHEETS_UPDATE_SHEET_PROPERTIES", spreadsheetId=spreadsheet_id, updateSheetProperties=update_sheet_properties)

@mcp.tool()
async def update_spreadsheet_properties(ctx: Context, fields: str, properties: Dict, spreadsheet_id: str) -> Dict:
    """Update properties of a spreadsheet."""
    return await _execute_tool(ctx, "GOOGLESHEETS_UPDATE_SPREADSHEET_PROPERTIES", fields=fields, properties=properties, spreadsheetId=spreadsheet_id)

@mcp.tool()
async def aggregate_column_data(ctx: Context, operation: str, search_column: str, search_value: str, sheet_name: str, spreadsheet_id: str, target_column: str, case_sensitive: bool = True, has_header_row: bool = True, percentage_total: Optional[float] = None) -> Dict:
    """Searches for rows where a specific column matches a value and performs mathematical operations on data from another column."""
    return await _execute_tool(ctx, "GOOGLESHEETS_AGGREGATE_COLUMN_DATA", operation=operation, search_column=search_column, search_value=search_value, sheet_name=sheet_name, spreadsheet_id=spreadsheet_id, target_column=target_column, case_sensitive=case_sensitive, has_header_row=has_header_row, percentage_total=percentage_total)

@mcp.tool()
async def batch_get_spreadsheet(ctx: Context, spreadsheet_id: str, ranges: Optional[List[str]] = None) -> Dict:
    """Retrieves data from specified cell ranges in a google spreadsheet."""
    return await _execute_tool(ctx, "GOOGLESHEETS_BATCH_GET", spreadsheet_id=spreadsheet_id, ranges=ranges)

@mcp.tool()
async def batch_update_values_by_data_filter(ctx: Context, data: List[Dict], spreadsheet_id: str, value_input_option: str, include_values_in_response: Optional[bool] = None, response_date_time_render_option: str = "SERIAL_NUMBER", response_value_render_option: str = "FORMATTED_VALUE") -> Dict:
    """Update values in ranges matching data filters."""
    return await _execute_tool(ctx, "GOOGLESHEETS_BATCH_UPDATE_VALUES_BY_DATA_FILTER", data=data, spreadsheetId=spreadsheet_id, valueInputOption=value_input_option, includeValuesInResponse=include_values_in_response, responseDateTimeRenderOption=response_date_time_render_option, responseValueRenderOption=response_value_render_option)

@mcp.tool()
async def clear_spreadsheet_values(ctx: Context, range: str, spreadsheet_id: str) -> Dict:
    """Clears cell content from a specified a1 notation range."""
    return await _execute_tool(ctx, "GOOGLESHEETS_CLEAR_VALUES", range=range, spreadsheet_id=spreadsheet_id)

@mcp.tool()
async def create_chart(ctx: Context, chart_type: str, data_range: str, spreadsheet_id: str, background_blue: Optional[float] = None, background_green: Optional[float] = None, background_red: Optional[float] = None, legend_position: str = "BOTTOM_LEGEND", sheet_id: Optional[int] = None, subtitle: Optional[str] = None, title: Optional[str] = None, x_axis_title: Optional[str] = None, y_axis_title: Optional[str] = None) -> Dict:
    """Create a chart in a google sheets spreadsheet."""
    return await _execute_tool(ctx, "GOOGLESHEETS_CREATE_CHART", chart_type=chart_type, data_range=data_range, spreadsheet_id=spreadsheet_id, background_blue=background_blue, background_green=background_green, background_red=background_red, legend_position=legend_position, sheet_id=sheet_id, subtitle=subtitle, title=title, x_axis_title=x_axis_title, y_axis_title=y_axis_title)

@mcp.tool()
async def create_google_sheet(ctx: Context, title: str) -> Dict:
    """Creates a new google spreadsheet in google drive."""
    return await _execute_tool(ctx, "GOOGLESHEETS_CREATE_GOOGLE_SHEET1", title=title)

@mcp.tool()
async def create_spreadsheet_column(ctx: Context, sheet_id: int, spreadsheet_id: str, inherit_from_before: Optional[bool] = None, insert_index: Optional[int] = None) -> Dict:
    """Creates a new column in a google spreadsheet."""
    return await _execute_tool(ctx, "GOOGLESHEETS_CREATE_SPREADSHEET_COLUMN", sheet_id=sheet_id, spreadsheet_id=spreadsheet_id, inherit_from_before=inherit_from_before, insert_index=insert_index)

@mcp.tool()
async def create_spreadsheet_row(ctx: Context, sheet_id: int, spreadsheet_id: str, inherit_from_before: Optional[bool] = None, insert_index: Optional[int] = None) -> Dict:
    """Inserts a new, empty row into a specified sheet."""
    return await _execute_tool(ctx, "GOOGLESHEETS_CREATE_SPREADSHEET_ROW", sheet_id=sheet_id, spreadsheet_id=spreadsheet_id, inherit_from_before=inherit_from_before, insert_index=insert_index)

@mcp.tool()
async def find_worksheet_by_title(ctx: Context, spreadsheet_id: str, title: str) -> Dict:
    """Finds a worksheet by its exact, case-sensitive title."""
    return await _execute_tool(ctx, "GOOGLESHEETS_FIND_WORKSHEET_BY_TITLE", spreadsheet_id=spreadsheet_id, title=title)

@mcp.tool()
async def format_cell(ctx: Context, end_column_index: int, end_row_index: int, spreadsheet_id: str, start_column_index: int, start_row_index: int, worksheet_id: int, blue: float = 0.9, bold: Optional[bool] = None, font_size: int = 10, green: float = 0.9, italic: Optional[bool] = None, red: float = 0.9, strikethrough: Optional[bool] = None, underline: Optional[bool] = None) -> Dict:
    """Applies text and background cell formatting to a specified range."""
    return await _execute_tool(ctx, "GOOGLESHEETS_FORMAT_CELL", end_column_index=end_column_index, end_row_index=end_row_index, spreadsheet_id=spreadsheet_id, start_column_index=start_column_index, start_row_index=start_row_index, worksheet_id=worksheet_id, blue=blue, bold=bold, fontSize=font_size, green=green, italic=italic, red=red, strikethrough=strikethrough, underline=underline)

@mcp.tool()
async def get_sheet_names(ctx: Context, spreadsheet_id: str) -> Dict:
    """Lists all worksheet names from a specified google spreadsheet."""
    return await _execute_tool(ctx, "GOOGLESHEETS_GET_SHEET_NAMES", spreadsheet_id=spreadsheet_id)

@mcp.tool()
async def lookup_spreadsheet_row(ctx: Context, query: str, spreadsheet_id: str, case_sensitive: Optional[bool] = None, range: Optional[str] = None) -> Dict:
    """Finds the first row in a google spreadsheet where a cell's content matches the query."""
    return await _execute_tool(ctx, "GOOGLESHEETS_LOOKUP_SPREADSHEET_ROW", query=query, spreadsheet_id=spreadsheet_id, case_sensitive=case_sensitive, range=range)

@mcp.tool()
async def search_spreadsheets(ctx: Context, query: Optional[str] = None, created_after: Optional[str] = None, include_trashed: Optional[bool] = None, max_results: int = 10, modified_after: Optional[str] = None, order_by: str = "modifiedTime desc", shared_with_me: Optional[bool] = None, starred_only: Optional[bool] = None) -> Dict:
    """Search for google spreadsheets using various filters."""
    return await _execute_tool(ctx, "GOOGLESHEETS_SEARCH_SPREADSHEETS", query=query, created_after=created_after, include_trashed=include_trashed, max_results=max_results, modified_after=modified_after, order_by=order_by, shared_with_me=shared_with_me, starred_only=starred_only)

@mcp.tool()
async def sheet_from_json(ctx: Context, sheet_json: List[Dict], sheet_name: str, title: str) -> Dict:
    """Creates a new google spreadsheet and populates its first worksheet from JSON."""
    return await _execute_tool(ctx, "GOOGLESHEETS_SHEET_FROM_JSON", sheet_json=sheet_json, sheet_name=sheet_name, title=title)

if __name__ == "__main__":
    host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_SERVER_PORT", 9015))
    
    print(f"Starting GSheets MCP Server on http://{host}:{port}")
    mcp.run(transport="sse", host=host, port=port)