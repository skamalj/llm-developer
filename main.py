from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langgraph.prebuilt import ToolNode
from devops_agent import route_to_devops_agent
from verifier import route_to_verifier_agent
from developer_agent import route_to_developer_agent
from tester_agent import route_to_tester_agent
import traceback
from langchain.schema import SystemMessage

# Initialize the Supervisor's model
#supervisor_model = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
supervisor_model = ChatOpenAI(model="gpt-4", temperature=0)

# Initialize the ToolNode with the route to the DevOps agent
tool_node = ToolNode(tools=[route_to_devops_agent, route_to_verifier_agent, route_to_developer_agent, route_to_tester_agent])

# Bind tools to the supervisor model
supervisor = supervisor_model.bind_tools([route_to_devops_agent, route_to_verifier_agent, route_to_developer_agent, route_to_tester_agent])

# Function to determine the next state
def should_continue(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "team"
    return END

# Function to call the supervisor model
def call_supervisor_model(state: MessagesState):
    
    with open("supervisor_prompt.txt", "r", encoding="utf-8") as file:
        system_message = file.read()
        messages = state["messages"]
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            # Create and prepend the system message
            system_msg = SystemMessage(content=system_message)
            messages.insert(0, system_msg)
        response = supervisor.invoke(messages)
        return {"messages": [response]}

# Define the Supervisor Agent's StateGraph
supervisor_graph = StateGraph(MessagesState)

# Add nodes to the graph
supervisor_graph.add_node("supervisor", call_supervisor_model)
supervisor_graph.add_node("team", tool_node)

# Define edges between nodes
supervisor_graph.add_edge(START, "supervisor")
supervisor_graph.add_conditional_edges(
    "supervisor", should_continue, ["team", END]
)
supervisor_graph.add_edge("team", "supervisor")

# Compile the Supervisor Agent
supervisor_agent = supervisor_graph.compile()
    
# Example task for the supervisor agent
input_message = {
    "messages": [
        ("human", """
Write a code for  instructions specified in /home/kamal/dev/llms/langgraph/llm-developer/BaseStore_def.txt. Project will use python 3.10
Project directory to be used for dvelopment and testing: /home/kamal/dev/cosmos_store_test
Developer should create source codee in <project_dir>/src
Tester should write test cases in <project_dr>/tests and use pytest to write test cases.
         """)
    ]
}

# Stream and process the output
try:
    for chunk in supervisor_agent.stream(input_message, subgraphs=True, stream_mode="values" , config={"recursion_limit": 30}):
        # Extract the last message from the chunk
        message = chunk[1]["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            try:
                message.pretty_print()
            except AttributeError:
                print("\nOutput:", message)
except ValueError as e:
    print(f"Error occurred: {traceback.print_exc()}")
