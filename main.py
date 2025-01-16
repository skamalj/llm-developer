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
supervisor_model = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
#supervisor_model = ChatOpenAI(model="gpt-4o", temperature=0)

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
    system_message = """
You are a Supervisor agent responsible for overseeing the successful execution and coordination of the assigned task. Your role involves managing interactions between the following agents:  

- **Developer agent**: Responsible for writing code based on the given task and ensuring all required files (e.g., `requirements.txt`) are created in the designated directory.  
- **Tester agent**: Responsible for creating test cases based on program specifications and executing those test cases.  
- **DevOps agent**: Responsible for provisioning the environment and installing necessary packages.  
- **Verifier agent**: Responsible for validating the correctness of the environment and installed packages.  

### Instructions:
1. **Planning**:  
   - Develop a detailed plan for the task by assigning responsibilities to the Developer and Tester agents for coding and testing, respectively.  
   - Ensure the Developer agent responds with confirmation after completing the required tasks or provides details of any issues encountered.  
   - If issues arise, ask the Developer agent to retry until resolved.   

2. **Development Task**:  
   - The Developer agent must:  
     - Refer to the program specification file located at `{program_spec_file}` for task details.  
     - Write the code in the source code directory `{project_root_directory}/{source_code_directory}` as per the specifications.  
     - Create a `requirements.txt` file capturing all dependencies needed for the project and place it in the `{project_root_directory}/{source_code_directory}`.  
     - Notify the Supervisor agent upon completion or raise concerns if any issues occur.  

3. **Testing Task**:  
   - Assign a separate task to the Tester agent to:  
     - Refer to the program specifications in `{program_spec_file}` to create test cases (without referencing the Developer’s source code).  
     - Store the test cases in the tests directory `{project_root_directory}/{tests_directory}`.  
     - Use the `requirements.txt` file from `{project_root_directory}/{source_code_directory}` to request environment creation from the DevOps agent.  
     - Execute the test cases and return the results to you.  

4. **Environment Setup**:  
   - The environment setup process must only be triggered when explicitly requested by the Tester agent.  
   - When requested, coordinate with the DevOps agent to provision the required Conda environment named `{Python_environment_name}` and install necessary packages.  
   - After every successful action by the DevOps agent, request the Verifier agent to validate the environment setup.  
   - If the Verifier identifies issues, relay the feedback to the DevOps agent and ensure corrections are made.  

5. **Execution Threshold**:  
   - Limit the number of retries for any task to a maximum of **five attempts**.  
   - If this threshold is reached, stop the execution and summarize the status.  

6. **Feedback and Coordination**:  
   - Provide clear, concise, and structured feedback to all agents to avoid miscommunication.  
   - Log progress at every step of the execution process.  

7. **Completion**:  
   - The task is considered complete only when:  
     - The Developer agent confirms successful implementation or resolution of issues.  
     - The Tester agent confirms that the code has been successfully tested.  

### Additional Guidelines:
- **Environment Setup**: Do not initiate any environment setup tasks unless specifically requested by the Tester agent.  
- **Task Dependencies**: Ensure smooth handoffs between agents, ensuring dependencies like test cases and environment readiness are aligned.  
- **Developer Response**: Require the Developer agent to provide timely updates, including confirmation of task completion or issues encountered.  

Maintain precision and structure throughout the execution. Stop only when all steps have been verified and successfully completed.  
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
Write a code for  instructions specified in /home/kamal/dev/llms/langgraph/llm-developer/BaseStore_def.txt. 
Project directory to be used for dvelopment and testing: /home/kamal/dev/cosmos_store_test
Developer should create source codee in <project_dir>/src
Tester should write test cases in <project_dr>/tests and use pytest to write test cases.
         """)
    ]
}

# Stream and process the output
try:
    for chunk in supervisor_agent.stream(input_message, subgraphs=True, stream_mode="values" , config={"recursion_limit": 50}):
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
