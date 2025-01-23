# @! using tools described in the incldued code include=tools.py instantiate anthropic model for langraph and langchain provider=anthropic

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from tools import  execute_os_commands, execute_conda_env_commands
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from typing import Annotated
from utils import handle_tool_calls


# Initialize the Anthropic model
#model = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Configure tools
tools = [execute_os_commands, execute_conda_env_commands]

# Create tool node for LangGraph
tool_node = ToolNode(tools=tools)

# @! bind tools to model
model_with_tools  = model.bind_tools(tools)

# Create a state graph for the agent
def should_continue(state: MessagesState) -> str:
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return 'tools'
    return END

def call_model(state: MessagesState):
    messages = handle_tool_calls(state["messages"])
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

envflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
envflow.add_node("agent", call_model)
envflow.add_node("tools", tool_node)

envflow.add_edge(START, "agent")
envflow.add_conditional_edges("agent", should_continue, ["tools", END])
envflow.add_edge("tools", "agent")

env_agent = envflow.compile()

@tool
def route_to_devops_agent(command_str: str):
    """
    Routes a command to the devops agent
    Handles tasks related to environment setup, CI/CD, and infrastructure management.
    command_str: Command to tool to execute in natural language 
    """
    response = env_agent.invoke({"messages": [{"role": "human", "content": command_str}]})
    return response["messages"][-1].content