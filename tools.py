from langchain_core.tools import tool
import subprocess

def execute_command(command: str, capture_output: bool = False) -> str:
    """
    Executes a shell command and returns its output as a string.

    :param command: A string representing the shell command to execute.
    :param capture_output: Boolean to indicate if you need commands output along with status or not.  Command outputs can be expensive to capture, so be cautious and use only when needed.
    :return: The standard output of the command as a stripped string.
    """
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE if capture_output else subprocess.DEVNULL, text=True, timeout=60, stderr=subprocess.PIPE)
    return result

@tool
def execute_os_commands(commands: list, capture_output: bool = False) -> list:
    """
    Executes a list of shell commands and returns their results.
    If any command needs interactive input, then use appropriate flags to handle it or return without executing it. These commmands will error out with Timeout waiting ofr human input, which will never happen.
    :param commands: A list of command strings to be executed.
    :param capture_output: Boolean to indicate if you need commands output along with status or not.  Command outputs can be expensive to capture, so be cautious and use only when needed.
    :return: A list of tuples containing the command and its execution result.
    """
    return [(command, execute_command(command, capture_output)) for command in commands]

@tool 
def execute_conda_env_commands(env_name: str, commands: list, capture_output: bool = False) -> list:
    """
    Execute list of commands which needs to be executed in a particular conda environment
    :param env_name: Environment in which commands are to be executed
    :param capture_output: Boolean to indicate if you need commands output along with status or not.  Command outputs can be expensive to capture, so be cautious and use only when needed.
    :return: A list of tuples containing the command and its execution result.
    """
    env_prefix = f'conda run -n {env_name} '
    return [(command, execute_command(env_prefix + command, capture_output)) for command in commands]

@tool
def save_file(file_path: str, content: str) -> str:
    """
    Saves the given content to the specified file.

    :param file_path: Full path, including base directory,  of the file where the content should be saved.
    :param content: The content to save in the file.
    :return: A message indicating success or the error encountered.
    """
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        return f"File saved successfully at {file_path}"
    except Exception as e:
        return f"Error saving file at {file_path}: {str(e)}"

@tool
def read_file(file_path: str) -> str:
    """
    Reads a given file and returns it's content
    :param file_path: Full path of file to read
    :return: Files content
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        return str(e)



