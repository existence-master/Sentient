from fastmcp import FastMCP

mcp = FastMCP(name="MyCustomServer", instructions="This server calculates BMI.")

@mcp.tool()
def calculate_bmi(weight: float, height: float) -> float:
    """Calculate BMI given weight (kg) and height (m)."""
    return weight / (height ** 2)

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
