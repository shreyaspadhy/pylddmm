import subprocess
import shlex


def run_bash_command(command):
    """
        Simple fn that runs a bash command through Python with real-time 
        outputs

        Parameters
        ----------
        command: str
            bash command
    """
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc
