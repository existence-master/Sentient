from fastmcp import FastMCP, Context

mcp = FastMCP(name="MyCustomServer", instructions="This server calculates BMI.")

@mcp.tool
async def calculate_bmi(weight: float, height: float, ctx: Context | None=None) -> float:
    """Calculate BMI given weight (kg) and height (m)."""
     # Get HTTP request
    request = ctx.get_http_request()
    
    # Extract user context from headers
    user_id = request.headers.get("X-User-ID", "unknown")
    print(f"Calculating BMI for user {user_id} with weight {weight} kg and height {height} m")
    return weight / (height ** 2)

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)