import subprocess
import os
import venv
from pathlib import Path
from tempfile import NamedTemporaryFile
from langchain_core.tools import tool

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # adjust if needed
VENV_DIR = BASE_DIR / "venv"
DOCKER_IMAGE_NAME = "python-sandbox"


def is_docker_available() -> bool:
    """Check if Docker is installed and running."""
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=3)
        return result.returncode == 0
    except Exception:
        return False


def ensure_venv():
    """Create persistent virtual environment and install pip if needed."""
    if not VENV_DIR.exists():
        print("Creating persistent virtual environment...")
        venv.create(VENV_DIR, with_pip=True)
        python_bin = get_python_bin()
        subprocess.run([python_bin, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    else:
        python_bin = get_python_bin()
    return python_bin


def get_python_bin() -> str:
    """Return path to python executable in venv."""
    return str(VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))


def write_temp_code(code: str) -> str:
    """Write user code to a temporary file."""
    with NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        return f.name


def ensure_docker_image():
    """Check if the prebuilt Docker image exists, build if not."""
    try:
        result = subprocess.run(
            ["docker", "images", "-q", DOCKER_IMAGE_NAME],
            capture_output=True,
            text=True,
            check=True
        )
        if not result.stdout.strip():
            print(f"Docker image '{DOCKER_IMAGE_NAME}' not found. Building it now...")
            dockerfile = """
            FROM python:3.12-slim
            RUN pip install --quiet --upgrade pip
            WORKDIR /app
            """
            with NamedTemporaryFile("w", suffix=".Dockerfile", delete=False) as f:
                f.write(dockerfile)
                dockerfile_path = f.name
            subprocess.run(
                ["docker", "build", "-t", DOCKER_IMAGE_NAME, "-f", dockerfile_path, "."],
                check=True
            )
            os.remove(dockerfile_path)
        else:
            print(f"Using existing Docker image '{DOCKER_IMAGE_NAME}'")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Docker image setup failed: {e}")


def run_in_docker(code: str, timeout: int) -> str:
    """Run plain Python code inside a prebuilt Docker container."""
    ensure_docker_image()
    code_path = write_temp_code(code)
    try:
        print("Running code in Docker container...")
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{code_path}:/tmp/script.py",
            DOCKER_IMAGE_NAME,
            "python", "/tmp/script.py"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = result.stdout.strip()
        error = result.stderr.strip()

        if result.returncode == 0:
            return "Success " +  output or "Success (no output)"
        else:
            return f"Error (Exit code {result.returncode}):\n{error or output}"
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out"
    finally:
        os.remove(code_path)


def run_in_venv(code: str, timeout: int) -> str:
    """Run plain Python code inside the persistent venv."""
    print("Running code in persistent virtual environment...")
    python_bin = ensure_venv()
    code_path = write_temp_code(code)
    try:
        cmd = [python_bin, code_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env={"PYTHONPATH": str(BASE_DIR)})
        output = result.stdout.strip()
        error = result.stderr.strip()

        if result.returncode == 0:
            return "Success " +  output or "Success (no output)"
        else:
            return f"Error (Exit code {result.returncode}):\n{error or output}"
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out"
    finally:
        os.remove(code_path)


@tool
def run_in_sandbox_tool(code: str, timeout: int = 5) -> str:
    """
    Run Python code safely: Docker if available, otherwise persistent venv.

    IMPORTANT:
    - The code must be directly executable and produce output when run.
    - If the code only contains functions or classes, it should include a minimal runner,
      such as an "if __name__ == '__main__':" block or explicit function calls with print statements,
      so that run_in_sandbox can verify its output.
    - Returns the program output if execution succeeds, or detailed errors if it fails.
    - Returns 'Success (no output)' if the code runs correctly but produces no output.
    - Returns 'Error: Execution timed out' if it exceeds the time limit.
    """

    if is_docker_available():
        try:
            return run_in_docker(code, timeout)
        except Exception as e:
            print(f"Docker failed: {e}, falling back to venv...")
    else:
        print("Docker not available, using persistent venv...")
    return run_in_venv(code, timeout)


def run_in_sandbox(code: str, timeout: int = 5) -> str:
    """Run Python code safely: Docker if available, otherwise persistent venv."""
    if is_docker_available():
        try:
            return run_in_docker(code, timeout)
        except Exception as e:
            print(f"Docker failed: {e}, falling back to venv...")
    else:
        print("Docker not available, using persistent venv...")
    return run_in_venv(code, timeout)
