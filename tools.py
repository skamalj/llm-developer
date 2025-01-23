from langchain_core.tools import tool
from utils import execute_command

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
    Executes a list of commands in a specified conda environment.

    :param env_name: The conda environment in which the commands are executed.
    :param commands: A list of commands to be executed.
    :param capture_output: Set to True to capture command output, useful where output needs to be analyzed. 
                           By default, the return code and error message (if any) are captured. So mostly default setting should work.
                           Enable verbose output only when necessary to avoid overhead.
    :return: A list of tuples, each containing the command and its execution result.
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
    

@tool
def ask_user_input(issue_summary: str) -> str:
    """
    Asks the user for help on issue which happens for more than 2 times.

    :param prompt: The question or statement to present to the user.
    :return: The user's input as a string.
    """
    return input(issue_summary)

@tool
def update_file_content(file_path: str, content: str, start_line: int = None, end_line: int = None) -> str:
    """
    Updates the content of a file at specified line range or appends if no range is given.

    :param file_path: Path to the file to be updated.
    :param content: Content to be inserted into the file.
    :param start_line: Optional; starting line number for content insertion.
    :param end_line: Optional; ending line number for content insertion.
    :return: Success message or error message if an exception occurs.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        if start_line is not None and end_line is not None:
            lines[start_line:end_line] = [content + '\n']
        elif start_line is not None:
            lines[start_line:start_line] = [content + '\n']
        else:
            lines.append(content + '\n')

        with open(file_path, 'w') as file:
            file.writelines(lines)

        return 'File updated successfully'
    except Exception as e:
        return f'Error updating file: {str(e)}'