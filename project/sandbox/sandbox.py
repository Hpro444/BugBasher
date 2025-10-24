import subprocess
import os
import venv
from pathlib import Path
from tempfile import NamedTemporaryFile
from langchain_core.tools import tool

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # adjust if needed
VENV_DIR = BASE_DIR / "venv"
DOCKER_IMAGE_NAME = "python-pytest-sandbox"


def is_docker_available() -> bool:
    """Check if Docker is installed and running."""
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=3)
        return result.returncode == 0
    except Exception:
        return False


def ensure_venv():
    """Create persistent virtual environment and install pytest if needed."""
    if not VENV_DIR.exists():
        print("Creating persistent virtual environment...")
        venv.create(VENV_DIR, with_pip=True)
        python_bin = get_python_bin()
        subprocess.run([python_bin, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([python_bin, "-m", "pip", "install", "pytest"], check=True)
    else:
        python_bin = get_python_bin()
    return python_bin


def get_python_bin() -> str:
    """Return path to python executable in venv."""
    return str(VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))


def write_temp_test(code: str, use_pytest: bool) -> str:
    """Wrap user code in a pytest test function and write to a temporary file."""
    with NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        if use_pytest:
            wrapped_code = (
                    "import pytest\n\n"
                    "def test_user_code():\n"
                    + "\n".join(f"    {line}" if line.strip() else "" for line in code.splitlines())
            )
            f.write(wrapped_code)
        else:
            f.write(code)
        return f.name


def ensure_docker_image():
    """Check if the prebuilt Docker image exists, build if not."""
    try:
        # Check if image exists
        result = subprocess.run(
            ["docker", "images", "-q", DOCKER_IMAGE_NAME],
            capture_output=True,
            text=True,
            check=True
        )
        if not result.stdout.strip():
            print(f"Docker image '{DOCKER_IMAGE_NAME}' not found. Building it now...")
            dockerfile = f"""
            FROM python:3.12-slim
            RUN pip install --quiet pytest
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


def run_in_docker(code: str, timeout: int, use_pytest: bool) -> str:
    """Run code inside a prebuilt Docker container."""
    ensure_docker_image()
    test_path = write_temp_test(code, use_pytest)
    try:
        print("Running code in Docker container...")
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{test_path}:/tmp/test.py",
            DOCKER_IMAGE_NAME,
            "pytest" if use_pytest else "python", "/tmp/test.py",
            "--maxfail=1", "-q", "--disable-warnings", "--tb=short",
            "--import-mode=importlib", "--no-header", "--no-summary"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return "Success"
        else:
            output = result.stdout.strip() or result.stderr.strip()
            return f"Error:\n{output}"
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out"
    finally:
        os.remove(test_path)


def run_in_venv(code: str, timeout: int, use_pytest: bool) -> str:
    """Run code inside the persistent venv using pytest."""
    print("Running code in persistent virtual environment...")
    python_bin = ensure_venv()
    test_path = write_temp_test(code, use_pytest)
    try:
        cmd = [
            python_bin, "-m", "pytest",
            "--maxfail=1", "-q", "--disable-warnings", "--tb=short",
            "--import-mode=importlib",
            "--no-header", "--no-summary",
            test_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env={"PYTHONPATH": str(BASE_DIR)})
        if result.returncode == 0:
            return "Success"
        else:
            output = result.stdout.strip() or result.stderr.strip()
            return f"Error:\n{output}"
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out"
    finally:
        os.remove(test_path)


@tool
def run_in_sandbox_tool(code: str, timeout: int = 5) -> str:
    """Run Python code safely: Docker if available, otherwise persistent venv."""
    if is_docker_available():
        try:
            return run_in_docker(code, timeout, True)
        except Exception as e:
            print(f"Docker failed: {e}, falling back to venv...")
    else:
        print("Docker not available, using persistent venv...")
    return run_in_venv(code, timeout, True)


def run_in_sandbox(code: str, timeout: int = 5) -> str:
    """Run Python code safely: Docker if available, otherwise persistent venv."""
    if is_docker_available():
        try:
            return run_in_docker(code, timeout, False)
        except Exception as e:
            print(f"Docker failed: {e}, falling back to venv...")
    else:
        print("Docker not available, using persistent venv...")
    return run_in_venv(code, timeout, False)
