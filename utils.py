import subprocess
from langchain_core.messages import ToolMessage

def execute_command(command: str, capture_output: bool = False) -> str:
    """
    Executes a shell command and returns its output as a string.

    :param command: A string representing the shell command to execute.
    :param capture_output: Boolean to indicate if you need commands output along with status or not.  Command outputs can be expensive to capture, so be cautious and use only when needed.
    :return: The standard output of the command as a stripped string.
    """
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE if capture_output else subprocess.DEVNULL, text=True, timeout=60, stderr=subprocess.PIPE)
    return result

def handle_tool_calls(history):
    # Get the last message
    message = history[-1]
    
    # Check if the last message is a ToolMessage and there's a preceding message
    if isinstance(message, ToolMessage) and len(history) > 1:
        # Get the message before the ToolMessage
        previous_message = history[-2]
        
        # Check if the previous message has tool_calls
        if hasattr(previous_message, "tool_calls") and previous_message.tool_calls:
            # Set args of the first tool_call to an empty dictionary
            previous_message.response_metadata = {}
            previous_message.usage_metadata = {}
            print("Updated args for tool_calls[0]:", previous_message.tool_calls)
    
    return history
