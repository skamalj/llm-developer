from typing import Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
import json

# Import the DevOps Agent or other sub-agents
from devops_agent import env_agent

# Initialize the Supervisor's model
supervisor_model = ChatOpenAI(model="gpt-4", temperature=0)

# Supervisor function
def supervisor(state: MessagesState) -> Command[Literal["devops_agent", END]]:
    """
    Supervisor agent decides the next step based on the current state.
    Routes to sub-agents or ends execution.
    """
    # Supervisor prompt
    system_message = [{
        "type": "system",
        "content": """
        You are a Supervisor Agent responsible for managing sub-agents in a multi-agent system.
        Your task is to analyze the current context and decide whether to invoke a sub-agent or terminate the process.

        ### Available Sub-Agents:
        - devops_agent: Handles tasks related to environment setup, CI/CD, and infrastructure management.

        ### Your Response:
        Provide a JSON object with the following keys:
        - "next_agent": The name of the agent to invoke or "__end__" to terminate the process.
        - "reason": A brief explanation of your decision.

        Example Response:
        
        {
            "next_agent": "devops_agent",
            "reason": "The task involves infrastructure setup, which is handled by the DevOps Agent."
        }
        
        """
    }]

    # Format the context for the LLM
    # Format the context for the LLM (convert to a readable string)
    messages = state["messages"]
    messages = system_message + messages
    
    #formatted_prompt = prompt.format(context=str(messages))

    # Invoke the Supervisor model
    response = supervisor_model.invoke(messages)
    parsed_response = json.loads(response.content)

    # Parse the JSON response
    next_agent = parsed_response.get("next_agent", "__end__")
    reason = parsed_response.get("reason", "No reason provided.")

    if next_agent == "__end__":
        # Log the reason for ending
        print(f"Supervisor decided to end: {reason}")
        return Command(goto=END)  # End the graph execution

    # Log the decision
    print(f"Supervisor routed to: {next_agent}, Reason: {reason}")

    # Route to a specific agent
    return Command(goto=next_agent)

# Define the Supervisor Agent's StateGraph
supervisor_graph = StateGraph(MessagesState)

# Add the supervisor node
supervisor_graph.add_node("supervisor", supervisor)

# Add the sub-agent node (DevOps Agent in this case)
supervisor_graph.add_node("devops_agent", env_agent)

# Define transitions between nodes
supervisor_graph.add_edge(START, "supervisor")  # Start with the supervisor
supervisor_graph.add_edge("devops_agent", "supervisor")  # Return to the supervisor after the DevOps Agent
supervisor_graph.add_edge("supervisor", END)  # Allow the supervisor to terminate the process

# Compile the Supervisor Agent
supervisor_agent = supervisor_graph.compile()

def extract_last_message(data):
    for item in data:
        if len(item) < 2:
            print(item)
            continue
        _, content = item
        if 'agent' in content and 'messages' in content['agent']:
            messages = content['agent']['messages']
            last_message = messages[-1]['text'] if 'text' in messages[-1] else None
            print(last_message)
        elif 'tools' in content and 'messages' in content['tools']:
            messages = content['tools']['messages']
            last_message = messages[-1]['content'] if 'content' in messages[-1] else None
            print(last_message)

# example with a single tool call
for chunk in supervisor_agent.stream(
    {"messages": [("human", "create a conda environment 'langgraph-store-cosmosdb' and verify?")]}, subgraphs=True
):
    # print chunk only if it is dictionary
    extract_last_message(chunk)
    