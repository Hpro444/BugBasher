# BugBasher: Automated Python Bug Fixing Agent

**BugBasher** is a **ReAct-based AI agent** built using **LangGraph** and **Ollama**, capable of automatically detecting and fixing bugs in Python code.  
It leverages reasoning and execution steps to iteratively identify, repair, and verify code correctness.

---

## 🚀 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/BugBasher.git
cd BugBasher
```

---

### 2️⃣ Install Dependencies

Install all Python dependencies with:

```bash
pip install -r requirements.txt
```

#### 🐳 Docker Recommendation

It is **recommended** to have **Docker** installed — it’s used to safely sandbox LLM-generated code and to isolate the evaluation process.

If **Docker** is not available, the system will automatically fall back to using a **persistent Python virtual environment (venv)** for sandboxed execution.

This behavior is handled internally by the `sandbox` module, which checks for Docker availability and gracefully switches to `venv` if needed.

---

### 3️⃣ Setup Ollama

If you don’t have **Ollama** installed, you can either install it directly or use Docker.

#### 💻 Local Installation

You can download Ollama from the official website:  
👉 [https://ollama.com/download](https://ollama.com/download)

#### 🐳 Using Docker

**For CPU:**

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

**For GPU:**

```bash
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

Then pull the desired model (for example, `qwen3:0.6b`):

```bash
docker exec -it ollama ollama pull qwen3:0.6b
```

For more details, visit the official Docker page:  
👉 [https://hub.docker.com/r/ollama/ollama](https://hub.docker.com/r/ollama/ollama)

---

### 4️⃣ Configure Ollama in `config.ini`

Depending on your setup, create a **`config.ini`** file in the project root and fill it as follows:

#### 🐳 If Using Docker

```ini
[Ollama]
ollama_url = http://host.docker.internal:11434
ollama_model = qwen3:0.6b
```

#### 💻 If Running Locally (No Docker)

```ini
[Ollama]
ollama_url = http://127.0.0.1:11434
ollama_model = qwen3:0.6b
```

#### 🌐 If Using a Remote Ollama Instance

```ini
[Ollama]
ollama_url = http://192.168.x.x:11434
ollama_model = qwen3:0.6b
```

---

## 🧠 Usage

You can use **BugBasher** either from the **Command Line Interface (CLI)** or through a **Streamlit-based GUI**.

---

### 💬 Command-Line Interface (CLI)

To launch the CLI, run:

```bash
python -m project.cli_app
```

#### 🐍 Fix a Python File

```bash
python -m project.cli_app fix <path_to_buggy_file.py>
```

**Optional arguments:**

- `--model` — override the model name from the config file
- `--url` — override the Ollama API URL

**Example:**

```bash
python -m project.cli_app fix examples/buggy_code.py --model qwen3:0.6b
```

This will:

1. Read the buggy file.
2. Send it to the **BugBashAgent** for debugging.
3. Save the fixed code as `<original_name>_fixed.py`.

---

#### 🧪 Evaluate the Model

```bash
python -m project.cli_app evaluate [--model <name>] [--url <url>] [--limit <n>]
```

**Options:**

- `--limit` — test only the first *n* problems from the **HumanEvalFix** benchmark.
- Results include total tests, passed cases, and **pass@1** accuracy.

**Example:**

```bash
python -m project.cli_app evaluate --model qwen3:0.6b --limit 20
```

---

### 🖥️ Graphical Interface (GUI)

You can also run BugBasher with a simple interactive GUI built using **Streamlit**:

```bash
python -m streamlit run project/gui_app.py
```

Then open your browser at:  
👉 [http://localhost:8501](http://localhost:8501)

#### ✨ Features

- **Fix Code Tab:** Paste buggy Python code and instantly get fixed, verified code.
- **Evaluate Model Tab:** Run the HumanEvalFix benchmark interactively with live progress and pass rate results.

---

## 🧩 Project Structure

```
BugBasher/
├── project/
│   ├── agent/
│   │   ├── model/
│   │   │   ├── __init__.py
│   │   │   └── BugBashAgent.py
│   │   ├── state/
│   │   │   ├── __init__.py
│   │   │   └── AgentState.py
│   ├── cleaners/
│   │   ├── __init__.py
│   │   └── LLMOutputCleaner.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── Configuration.py
│   ├── domain/
│   │   ├── __init__.py
│   │   └── EvalResult.py
│   ├── eval/
│   │   ├── dataset/
│   │   │   ├── __init__.py
│   │   │   └── HumanEvalFixDataset.py
│   │   ├── evaluator/
│   │   │   ├── __init__.py
│   │   │   └── BugFixEvaluator.py
│   │   └── __init__.py
│   ├── sandbox/
│   │   ├── __init__.py
│   │   └── sandbox.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── PythonBugFixerCLI.py
│   │   └── PythonBugFixerGUI.py
│   ├── cli_app.py
│   ├── __init__.py
│   └── gui_app.py
├── config.ini
├── requirements.txt
├── .gitignore
└── README.md
```

# 📊 Evaluation

BugBasher includes a fully automated **evaluation pipeline** that benchmarks the agent’s bug-fixing ability on the **HumanEvalFix** dataset — a collection of real-world buggy Python functions and their corresponding test suites.

---

## 🧠 How Evaluation Works

1. **Dataset Loading and Cleaning**  
   The evaluation system automatically loads the [BigCode HumanEvalPack](https://huggingface.co/datasets/bigcode/humanevalpack) dataset from Hugging Face.  
   A lightweight wrapper class, `HumanEvalFixDataset`, preprocesses the dataset by:
    - Combining the `declaration` and `buggy_solution` fields into a single prompt.
    - Attaching the associated `test` field.
    - Caching the cleaned dataset as a local pickle file (`humanevalfix_cleaned.pkl`) to avoid reloading on subsequent runs.

   Example structure of cleaned samples:
   ```python
   {
       "prompt": "<buggy_function_code>",
       "test": "<assertions_and_test_cases>"
   }
   ```

---

2. **Bug Fixing & Testing Loop**  
   The `BugFixEvaluator` class evaluates the model’s performance by iterating through the dataset:

    - For each sample:
        1. Sends the buggy code prompt to the **BugBashAgent**.
        2. Receives and cleans the LLM’s fixed code using `LLOutputCleaner`.
        3. Generates complete test code by merging the fix and its associated tests.
        4. Executes the code in a **sandboxed environment** (Docker or `venv` fallback).
        5. Records whether the test passed or failed.

    - The evaluator tracks:
        - ✅ **Passed tests**
        - ❌ **Failed tests**
        - 🧮 **pass@1 accuracy** (first-attempt success rate)

---

3. **Sandboxed Execution**

   Each generated fix is executed **safely inside a sandbox**, preventing malicious or unstable code from affecting the host environment.  
   The sandbox automatically selects between:
    - 🐳 **Docker** — preferred for full isolation
    - 🐍 **Python venv** — fallback when Docker is unavailable

   This ensures secure, repeatable, and fair evaluation for all models.

---

## 🧾 Example Evaluation Command

Run the evaluation from the CLI:

```bash
python -m project.cli_app evaluate --model qwen3:0.6b --limit 20
```

or, for the full dataset:

```bash
python -m project.cli_app evaluate --model qwen3:8b
```

---

## 🧪 Example Results

| Model        | Passed | Total | Pass@1 | Score %   |
|--------------|--------|-------|--------|-----------|
| `qwen3:0.6b` | 9      | 164   | 0.0549 | **5.5%**  |
| `qwen3:8b`   | 153    | 164   | 0.9329 | **93.3%** |

---

## 🧩 Evaluation Output Format

Each evaluation produces an `EvalResult` object:

```json
{
  "passed": 153,
  "total": 164,
  "model_name": "qwen3:8b",
  "score_in_percentage": "93.3%",
  "score_pass1": 0.9329268292682927
}
```

### Why `qwen3:0.6b` Performed Poorly

The significantly lower performance of `qwen3:0.6b` (≈5.5% pass rate) compared to larger models like `qwen3:8b` (≈93.3%) is mainly due to its inability to consistently follow structured tool-use instructions and strict output requirements.

The `BugBashAgent` relies on a language model that can:

- Understand the system prompt and adhere to *strict output-only* rules.
- Correctly call the provided sandbox tool (`run_in_sandbox`) to test and verify its fixes.
- Maintain proper reasoning flow through multiple tool calls (up to 5 per debugging session).
- Return only valid, executable Python code — with no explanations or stray text.

However, `qwen3:0.6b` frequently violated these constraints:

- It often ignored the instruction to call `run_in_sandbox`, trying to return results directly instead of verifying fixes.
- It produced incomplete or non-executable code (e.g., missing functions or incorrect indentation).
- It mixed explanations with the code output, returning long descriptive text blocks followed by partial code — completely breaking the expected structure.
- It sometimes tried to call the tool as if it were a normal Python function (e.g., `run_in_sandbox("code")`) instead of using the structured tool-calling format expected by the LangGraph framework.
- In some cases, it even returned the *final* fixed code **wrapped inside a call to** `run_in_sandbox(...)`, which immediately failed in the evaluation phase because that function doesn’t exist in the isolated test environment.

To mitigate this, additional handling logic was introduced in the `_tool_code_fixer_node` of the agent.  
This node attempts to detect and repair malformed tool call structures by parsing the model’s output and reconstructing missing or incorrect `tool_calls` before execution. While this improved stability somewhat, the model’s unpredictable formatting and lack of consistency made it impossible to guarantee correct behavior in all cases.

In summary, the poor performance of `qwen3:0.6b` is a result of:

- Limited model capacity for multi-step reasoning and structured tool adherence.
- Mixing explanations and code in output.
- Misusing tool calls (e.g., calling them as Python functions or embedding them in code).
- Returning code that cannot execute in the evaluation environment.
- Unreliable output formatting under strict system constraints.

Larger models like `qwen3:8b` demonstrate a much stronger ability to follow instructions, properly use tools, and consistently produce valid, working Python code that passes evaluation.
