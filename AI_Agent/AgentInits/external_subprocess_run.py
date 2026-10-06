import subprocess

class InitModel():

    def __init__(self):
        self.process_settings = {
            "text": False,
            "capture_output": False,
            "bufsize": -1,
            "executable": None,
            "stdin": None,
            "stdout": None,
            "stderr": None,
            "preexec_fn": None,
            "close_fds": True,
            "shell": False,
            "cwd": None,
            "env": None,
            "universal_newlines": None,
            "startupinfo": None,
            "creationflags": 0,
            "restore_signals": True,
            "start_new_session": False,
            "pass_fds": (),
            "check": False,
            "encoding": None,
            "errors": None,
            "input": None,
            "timeout": None,
            "user": None,
            "group": None,
            "extra_groups": None,
            "umask": -1,
            "pipesize": -1,
            "process_group": None
        }

        self.run_process()

    def run_process(self, command: str = None):
        config = self.process_settings

        if command is not None:
            try:
                subprocess.run(
                    [command],
                    bufsize=config["bufsize"],
                    executable=config["executable"],
                    stdin=config["stdin"],
                    stdout=config["stdout"],
                    stderr=config["stderr"],
                    preexec_fn=config["preexec_fn"],
                    close_fds=config["close_fds"],
                    shell=config["shell"],
                    cwd=config["cwd"],
                    env=config["env"],
                    universal_newlines=config["universal_newlines"],
                    startupinfo=config["startupinfo"],
                    creationflags=config["creationflags"],
                    restore_signals=config["restore_signals"],
                    start_new_session=config["start_new_session"],
                    pass_fds=config["pass_fds"],
                    capture_output=config["capture_output"],
                    check=config["check"],
                    encoding=config["encoding"],
                    errors=config["errors"],
                    input=config["input"],
                    text=config["text"],
                    timeout=config["timeout"],
                    user=config["user"],
                    group=config["group"],
                    extra_groups=config["extra_groups"],
                    umask=config["umask"],
                    pipesize=config["pipesize"],
                    process_group=config["process_group"]
                )
            except Exception as process_error:
                print(process_error)

    def info(self):
        """Reference for subprocess.run configuration used by this helper."""
