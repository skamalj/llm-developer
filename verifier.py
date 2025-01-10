# @! create a verifier agent, this will be send task to verify actions which devops will eprform,  similar to agent in included code include=devops_agent.py 

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.tools import tool
from tools import execute_os_commands, execute_conda_env_commands

# Initialize the Anthropic model for verifier
#model = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Configure tools
tools = [execute_os_commands, execute_conda_env_commands]

# Create tool node for LangGraph
tool_node = ToolNode(tools=tools)

# @! bind tools to model
verifier_model  = model.bind_tools(tools)

# Create a state graph for the verifier agent
def verifier_should_continue(state: MessagesState) -> str:
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return 'tools'
    return END

def call_verifier_model(state: MessagesState):
    messages = state["messages"]
    response = verifier_model.invoke(messages)
    return {"messages": [response]}

verifier_flow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
verifier_flow.add_node("verifier_agent", call_verifier_model)
verifier_flow.add_node("tools", ToolNode(tools=[execute_os_commands, execute_conda_env_commands]))

verifier_flow.add_edge(START, "verifier_agent")
verifier_flow.add_conditional_edges("verifier_agent", verifier_should_continue, ["tools", END])
verifier_flow.add_edge("tools", "verifier_agent")

verifier_agent = verifier_flow.compile()

@tool
def route_to_verifier_agent(command_str: str):
    """
    Routes a command to the verifier agent
    Handles tasks related to verification of environment created by devops agent.
    command_str: Command to tool to execute in natural language 
    """
    response = verifier_agent.invoke({"messages": [{"role": "human", "content": command_str}]})
    return response["messages"][-1]