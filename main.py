from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langgraph.prebuilt import ToolNode
from devops_agent import route_to_devops_agent
from verifier import route_to_verifier_agent
from developer_agent import route_to_developer_agent
import traceback
from langchain.schema import SystemMessage

# Initialize the Supervisor's model
#supervisor_model = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
supervisor_model = ChatOpenAI(model="gpt-4o", temperature=0)

# Initialize the ToolNode with the route to the DevOps agent
tool_node = ToolNode(tools=[route_to_devops_agent, route_to_verifier_agent, route_to_developer_agent])

# Bind tools to the supervisor model
supervisor = supervisor_model.bind_tools([route_to_devops_agent, route_to_verifier_agent, route_to_developer_agent])

# Function to determine the next state
def should_continue(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "devops_agent"
    return END

# Function to call the supervisor model
def call_supervisor_model(state: MessagesState):
    system_message = """
You are a supervisor responsible for ensuring the successful execution and coordination of given task. 
You will coordinate between the Developer agent (tasked with writing code for the given ask) , DevOps agent (tasked with creating the environment and installing necessary packages) 
and the Verifier agent (tasked with verifying if the environment and packages are set up correctly). 

Instructions:
1. Create a plan for the ask by creating tasks for developer agent for developing a code
2. Create a separate task for developer to ensure it test the developed code and has created test cases for it.
3. Developer must pass the test cases result back to you close the development task.
4. Developer will ask yout help for provisioning of conda environment. 
    - Ensure that the DevOps agent performs actions to create the environment and install required packages.
    - After every SUCCESSFUL action by the DevOps agent, route a verification request to the Verifier agent to check the environment's state.
    - If the Verifier agent identifies issues, relay the feedback to the DevOps agent and prompt it to resolve them.
5. Do not re-attempt any task more than 5 times.  In case this threshold is met, stop the execution and summarize your status.
6. Provide clear and concise feedback between the agents to avoid confusion.
7. Stop only when the Developer agent confirms that the code has been tested successful.

Be precise and structured in your instructions. Log progress at every step.
"""

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
        ("human", """
Write a code for  instructions specified in /home/kamal/dev/llms/langgraph/llm-developer/BaseStore_def.txt. Create the code in /home/kamal/dev/cosmos_store_test
         """)
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
    print(f"Error occurred: {traceback.print_exc()}")
