# @! using tools described in the incldued code include=tools.py instantiate anthropic model for langraph and langchain provider=anthropic

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from tools import  execute_os_commands, execute_conda_env_commands, save_file, read_file, ask_user_input
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, InjectedState
from langchain_core.tools import tool
from typing_extensions import Annotated
from langchain.schema import SystemMessage
from langgraph_checkpoint_cosmosdb import CosmosDBSaver



# Initialize the Anthropic model
#model = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
model = ChatOpenAI(model="gpt-4o", temperature=0)


# Configure tools
tools = [execute_os_commands, execute_conda_env_commands, save_file, read_file, ask_user_input]

# Create tool node for LangGraph
tool_node = ToolNode(tools=tools)

# @! bind tools to model
model_with_tools  = model.bind_tools(tools)
saver = CosmosDBSaver(database_name='builderdb', container_name='checkpoint')
# Create a state graph for the agent
def should_continue(state: MessagesState) -> str:
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return 'tools'
    return END

def call_model(state: MessagesState):

    with open("tester_prompt.txt", "r", encoding="utf-8") as file:
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

dev_agent = devflow.compile(checkpointer=saver)

@tool
def route_to_tester_agent(command_str: str, 
                          python_environment_name: str, 
                          source_code_directory: str, 
                          tests_directory: str, 
                          program_spec_file: str, 
                          project_root_directory: str,
                          state: Annotated[dict, InjectedState]) -> str:
    """
    Routes a command to the tester agent.
    Handles tasks related to creating test cases and executing them.
    
    command_str: Command to tool to execute in natural language 
    python_environment_name: Name of the Python environment to be used
    source_code_directory: Path to the directory where the source code is located
    tests_directory: Path to the directory where test cases are located
    program_spec_file: Path to the program specification file
    project_root_directory: Path to the root directory of the project
    state: Injected state to persist the project configuration
    """
    
    # Check if each value exists in state, if not, set them
    if "python_environment_name" not in state:
        state["python_environment_name"] = python_environment_name
    if "source_code_directory" not in state:
        state["source_code_directory"] = source_code_directory
    if "tests_directory" not in state:
        state["tests_directory"] = tests_directory
    if "program_spec_file" not in state:
        state["program_spec_file"] = program_spec_file
    if "project_root_directory" not in state:
        state["project_root_directory"] = project_root_directory
    
    # Use the state values
    python_environment_name = state.get("python_environment_name", '')
    source_code_directory = state.get("source_code_directory", '')
    tests_directory = state.get("tests_directory", '')
    program_spec_file = state.get("program_spec_file", '')
    project_root_directory = state.get("project_root_directory", '')
    
    # Adding relevant context to the command string
    command_str += f"Environment Name: {python_environment_name}, " \
                   f"Source Code Directory: {source_code_directory}, " \
                   f"Tests Directory: {tests_directory}, " \
                   f"Program Spec File: {program_spec_file}, " \
                   f"Project Root Directory: {project_root_directory}"

    config = {"configurable": {"thread_id": "225"}}
    # Send the command to the Tester agent
    response = dev_agent.invoke({"messages": [{"role": "human", "content": command_str}]},config)
    
    return response["messages"][-1].content
