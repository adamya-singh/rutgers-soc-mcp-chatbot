import os
import json
from typing import List
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from openai import OpenAI
from .utils.prompt import ClientMessage, convert_to_openai_messages
from .utils.tools import get_current_weather
from .utils.xml_agent import get_xml_system_prompt, parse_xml_response, execute_tool


load_dotenv(".env.local")

app = FastAPI()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


class Request(BaseModel):
    messages: List[ClientMessage]


available_tools = {
    "get_current_weather": get_current_weather,
}

def do_stream(messages: List[ChatCompletionMessageParam]):
    stream = client.chat.completions.create(
        messages=messages,
        model="gpt-4o",
        stream=True,
        tools=[{
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather at a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {
                            "type": "number",
                            "description": "The latitude of the location",
                        },
                        "longitude": {
                            "type": "number",
                            "description": "The longitude of the location",
                        },
                    },
                    "required": ["latitude", "longitude"],
                },
            },
        }]
    )

    return stream

def stream_text(messages: List[ChatCompletionMessageParam], protocol: str = 'data'):
    draft_tool_calls = []
    draft_tool_calls_index = -1

    stream = client.chat.completions.create(
        messages=messages,
        model="gpt-4o",
        stream=True,
        tools=[{
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather at a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {
                            "type": "number",
                            "description": "The latitude of the location",
                        },
                        "longitude": {
                            "type": "number",
                            "description": "The longitude of the location",
                        },
                    },
                    "required": ["latitude", "longitude"],
                },
            },
        }]
    )

    for chunk in stream:
        for choice in chunk.choices:
            if choice.finish_reason == "stop":
                continue

            elif choice.finish_reason == "tool_calls":
                for tool_call in draft_tool_calls:
                    yield '9:{{"toolCallId":"{id}","toolName":"{name}","args":{args}}}\n'.format(
                        id=tool_call["id"],
                        name=tool_call["name"],
                        args=tool_call["arguments"])

                for tool_call in draft_tool_calls:
                    tool_result = available_tools[tool_call["name"]](
                        **json.loads(tool_call["arguments"]))

                    yield 'a:{{"toolCallId":"{id}","toolName":"{name}","args":{args},"result":{result}}}\n'.format(
                        id=tool_call["id"],
                        name=tool_call["name"],
                        args=tool_call["arguments"],
                        result=json.dumps(tool_result))

            elif choice.delta.tool_calls:
                for tool_call in choice.delta.tool_calls:
                    id = tool_call.id
                    name = tool_call.function.name
                    arguments = tool_call.function.arguments

                    if (id is not None):
                        draft_tool_calls_index += 1
                        draft_tool_calls.append(
                            {"id": id, "name": name, "arguments": ""})

                    else:
                        draft_tool_calls[draft_tool_calls_index]["arguments"] += arguments

            else:
                yield '0:{text}\n'.format(text=json.dumps(choice.delta.content))

        if chunk.choices == []:
            usage = chunk.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens

            yield 'e:{{"finishReason":"{reason}","usage":{{"promptTokens":{prompt},"completionTokens":{completion}}},"isContinued":false}}\n'.format(
                reason="tool-calls" if len(
                    draft_tool_calls) > 0 else "stop",
                prompt=prompt_tokens,
                completion=completion_tokens
            )

def stream_text_xml(messages: List[ChatCompletionMessageParam], debug: bool = False):
    """Stream responses using XML parsing approach"""
    
    # Insert XML system prompt - replace any existing system message
    xml_messages = [{"role": "system", "content": get_xml_system_prompt()}]
    
    # Add user messages (skip any existing system messages)
    for msg in messages:
        if msg["role"] != "system":
            xml_messages.append(msg)
    
    try:
        # Get complete response (not streaming for XML parsing)
        response = client.chat.completions.create(
            messages=xml_messages,
            model="gpt-4o",
            stream=False,
        )
        
        response_content = response.choices[0].message.content
        
        # If debug mode, show raw XML
        if debug:
            debug_output = f"🔍 RAW XML RESPONSE:\n\n{response_content}\n\n" + "="*50 + "\n\n"
            yield f'0:{json.dumps(debug_output)}\n'
        
        # Parse XML response
        reasoning, tool_call = parse_xml_response(response_content)
        
        # Stream the reasoning
        if reasoning:
            if debug:
                parsed_output = f"📋 PARSED REASONING:\n\n{reasoning}\n\n"
                yield f'0:{json.dumps(parsed_output)}\n'
            else:
                yield f'0:{json.dumps(reasoning)}\n'
        
        # Handle tool call if present
        if tool_call:
            tool_call_id = "xml_tool_call_1"
            
            if debug:
                tool_debug = f"🔧 PARSED TOOL CALL:\n\n{json.dumps(tool_call, indent=2)}\n\n"
                yield f'0:{json.dumps(tool_debug)}\n'
            
            # Yield tool call start
            yield f'9:{{"toolCallId":"{tool_call_id}","toolName":"{tool_call["tool"]}","args":{json.dumps(tool_call["args"])}}}\n'
            
            # Execute tool
            tool_result = execute_tool(tool_call)
            
            # Yield tool result
            yield f'a:{{"toolCallId":"{tool_call_id}","toolName":"{tool_call["tool"]}","args":{json.dumps(tool_call["args"])},"result":{json.dumps(tool_result)}}}\n'
        
        # Yield completion
        usage = response.usage
        yield f'e:{{"finishReason":"stop","usage":{{"promptTokens":{usage.prompt_tokens},"completionTokens":{usage.completion_tokens}}},"isContinued":false}}\n'
        
    except Exception as e:
        # Handle errors gracefully
        error_msg = f"Error in XML processing: {str(e)}"
        yield f'0:{json.dumps(error_msg)}\n'
        yield f'e:{{"finishReason":"stop","usage":{{"promptTokens":0,"completionTokens":0}},"isContinued":false}}\n'

@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query('data'), mode: str = Query('function'), debug: bool = Query(False)):
    messages = request.messages
    openai_messages = convert_to_openai_messages(messages)

    if mode == 'xml':
        response = StreamingResponse(stream_text_xml(openai_messages, debug=debug))
    else:
        response = StreamingResponse(stream_text(openai_messages, protocol))
        
    response.headers['x-vercel-ai-data-stream'] = 'v1'
    return response
