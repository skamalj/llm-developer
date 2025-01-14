# @! using tools described in the incldued code include=tools.py instantiate anthropic model for langraph and langchain provider=anthropic

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from tools import  execute_os_commands, execute_conda_env_commands, save_file
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from typing import Annotated
from langchain.schema import SystemMessage



# Initialize the Anthropic model
#model = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Configure tools
tools = [execute_os_commands, execute_conda_env_commands, save_file]

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

    with open("developer_prompt.txt", "r", encoding="utf-8") as file:
        system_message = file.read()

        messages = state["messages"]
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            # Create and prepend the system message
            system_msg = SystemMessage(content=system_message)
            messages.insert(0, system_msg)
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

devflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
devflow.add_node("agent", call_model)
devflow.add_node("tools", tool_node)

devflow.add_edge(START, "agent")
devflow.add_conditional_edges("agent", should_continue, ["tools", END])
devflow.add_edge("tools", "agent")

dev_agent = devflow.compile()

@tool
def route_to_developer_agent(command_str: str, work_dir: str):
    """
    Routes a command to the developer agent
    Handles tasks related to code development and testing.
    command_str: Command to tool to execute in natural language 
    work_dir: Work directory for this project
    """
    response = dev_agent.invoke({"messages": [{"role": "human", "content": f'{command_str}. Your work directory is {work_dir}'}]})
    return response["messages"][-1].content