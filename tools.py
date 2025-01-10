from langchain_core.tools import tool
import subprocess

def execute_command(command: str) -> str:
    """
    Executes a shell command and returns its output as a string.

    :param command: A string representing the shell command to execute.
    :return: The standard output of the command as a stripped string.
    """
    result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=20)
    return result.stdout.strip()

@tool
def execute_commands(commands: list) -> list:
    """
    Executes a list of shell commands and returns their results.
    If any command needs interactive input, then use appropriate flags to handle it or return without executing it. These commmands will error out with Timeout waiting ofr human input, which will never happen.
    :param commands: A list of command strings to be executed.
    :return: A list of tuples containing the command and its execution result.
    """
    return [(command, execute_command(command)) for command in commands]





