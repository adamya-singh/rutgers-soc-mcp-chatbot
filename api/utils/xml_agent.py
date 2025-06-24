import json
import verifiers as vf
from typing import Dict, Any, Tuple, Optional
from .tools import get_current_weather

# Setup XML parser
parser = vf.XMLParser(fields=["reasoning", ("tool", "answer")], answer_field="answer")

# Available tools mapping
available_tools = {
    "get_current_weather": get_current_weather,
}

def get_xml_system_prompt() -> str:
    """Generate the system prompt for XML-based responses"""
    return f"""
You have access to a 'get_current_weather' tool. **Always think step-by-step before calling a tool.**

Args:
- latitude: float (latitude of the location)
- longitude: float (longitude of the location)

Respond in the following XML format:
{parser.get_format_str()}

For tool calls, return a JSON object inside the 'tool' section with the following fields:
- tool: str (e.g. "get_current_weather")
- args: dict (e.g. {{"latitude": 40.7128, "longitude": -74.0060}})

Always include your reasoning in the 'reasoning' section before making any tool calls.
If no tool is needed, leave the 'tool' section empty or put null.
"""

def parse_xml_response(response: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Parse XML response and return reasoning and tool call if present"""
    try:
        parsed = parser.parse(response)
        reasoning = parsed.reasoning if hasattr(parsed, 'reasoning') else ""
        tool_call = parsed.tool if hasattr(parsed, 'tool') else None
        
        if tool_call and isinstance(tool_call, str) and tool_call.strip():
            try:
                tool_call = json.loads(tool_call)
            except json.JSONDecodeError:
                print(f"Failed to parse tool call JSON: {tool_call}")
                tool_call = None
        elif not tool_call or (isinstance(tool_call, str) and not tool_call.strip()):
            tool_call = None
                
        return reasoning, tool_call
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return response, None

def execute_tool(tool_call: Dict[str, Any]) -> Any:
    """Execute a tool call and return the result"""
    tool_name = tool_call.get("tool")
    tool_args = tool_call.get("args", {})
    
    if tool_name in available_tools:
        try:
            return available_tools[tool_name](**tool_args)
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}
    else:
        return {"error": f"Unknown tool: {tool_name}"} 