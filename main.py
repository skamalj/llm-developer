from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langgraph.prebuilt import ToolNode
from devops_agent import route_to_devops_agent

# Initialize the Supervisor's model
supervisor_model = ChatOpenAI(model="gpt-4", temperature=0)

# Initialize the ToolNode with the route to the DevOps agent
tool_node = ToolNode(tools=[route_to_devops_agent])

# Bind tools to the supervisor model
supervisor = supervisor_model.bind_tools([route_to_devops_agent])

# Function to determine the next state
def should_continue(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "devops_agent"
    return END

# Function to call the supervisor model
def call_supervisor_model(state: MessagesState):
    messages = state["messages"]
    response = supervisor.invoke(messages)
    return {"messages": [response]}

# Define the Supervisor Agent's StateGraph
supervisor_graph = StateGraph(MessagesState)

# Add nodes to the graph
supervisor_graph.add_node("supervisor", call_supervisor_model)
supervisor_graph.add_node("devops_agent", tool_node)

# Define edges between nodes
supervisor_graph.add_edge(START, "supervisor")
supervisor_graph.add_conditional_edges(
    "supervisor", should_continue, ["devops_agent", END]
)
supervisor_graph.add_edge("devops_agent", "supervisor")

# Compile the Supervisor Agent
supervisor_agent = supervisor_graph.compile()

# Example task for the supervisor agent
input_message = {
    "messages": [
        ("human", "Create a conda environment 'langgraph-store-cosmosdb' and install langgraph in this environment. Verify the package is installed successfully and provide proof with a directory listing using 'ls'.")
    ]
}

# Stream and process the output
try:
    for chunk in supervisor_agent.stream(input_message, subgraphs=True, stream_mode="values"):
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
    print(f"Error occurred: {e}")
