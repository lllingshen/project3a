"""One subprocess with bounded startup and request timeouts (stdlib only)."""
import json
import os
from pathlib import Path
import select
import subprocess


class WorkerClient:
    def __init__(self, python, model_path, eagle_path):
        env = os.environ.copy()
        # ROS's PYTHONPATH must not inject its Python packages into conda.
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        self.process = subprocess.Popen(
            [python, "-u", str(Path(__file__).with_name("model_worker.py")),
             "--model-path", model_path, "--eagle-path", eagle_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env,
        )

    def read(self, timeout):
        ready, _, _ = select.select([self.process.stdout], [], [], timeout)
        if not ready:
            self.close()
            raise TimeoutError("LocateAnything worker timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("LocateAnything worker exited before responding")
        result = json.loads(line)
        if result.get("status") == "error":
            raise RuntimeError(result.get("error", "worker failure"))
        return result

    def infer(self, request, timeout):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return self.read(timeout)

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                # A second Ctrl-C can arrive while ROS launch is shutting
                # down the process group. Finish cleanup without a traceback.
                self.process.kill()
                try:
                    self.process.wait(timeout=3)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    pass
