from tqdm import tqdm
from project.sandbox import run_in_sandbox
from project.agent.model import BugBashAgent
from project.eval.dataset import HumanEvalFixDataset
from typing import Optional
import textwrap
from project.domain import EvalResult
from project.cleaners import LLOutputCleaner

class BugFixEvaluator:

    def __init__(self, agent: Optional[BugBashAgent] = None, dataset: Optional[HumanEvalFixDataset] = None):
        self._agent = agent if agent is not None else BugBashAgent()
        self._dataset = dataset if dataset is not None else HumanEvalFixDataset()
        self._code_cleaner = LLOutputCleaner()



    @staticmethod
    def _generate_test_code(llm_code: str, test_code: str) -> str:
        code = f"""
    {llm_code}

    {test_code}

    """
        # Dedent to remove unwanted leading spaces
        return textwrap.dedent(code).strip() + "\n"


    def evaluate(self, num_tests: Optional[int] = None):
        """
        Evaluates the LLM on a subset (or full set) of the dataset.
        Runs each generated fix in a sandbox and counts passing tests.
        """
        dataset = self._dataset.dataset
        num_tests = num_tests or len(dataset)

        passed = 0
        total = min(num_tests, len(dataset))

        print(f"Running evaluation on {total} samples...")

        for i in tqdm(range(total), desc="Evaluating", unit="test"):
            sample = dataset[i]
            prompt = sample["prompt"]
            test_code = sample["test"]

            # Ask the agent to fix the bug
            llm_output = self._agent.invoke(prompt)
            llm_code = self._code_cleaner.get_code_from_llm_output(llm_output)

            # Build the complete test code
            code_to_run = self._generate_test_code(llm_code, test_code)

            # Run inside sandbox
            result = run_in_sandbox(code_to_run)
            # Simple pass/fail detection
            if "Success" in result:
                passed += 1


        output = EvalResult(passed=passed,total=total,score_in_percentage=f"{passed / total:.1%}",score_pass1=passed / total,model_name=self._agent.get_model())


        return output

    def evaluate_for_gui(self, progress_bar=None, progress_text=None, status_box=None, num_tests: Optional[int] = None):
        """
        Runs evaluation with live progress updates for Streamlit.

        Parameters
        ----------
        progress_bar : st.progress
            Streamlit progress bar to update.
        progress_text : st.empty
            Streamlit text placeholder for percent updates.
        status_box : st.empty
            Optional Streamlit box for live status output.
        num_tests : Optional[int]
            Number of tests to run (defaults to full dataset).
        """
        dataset = self._dataset.dataset
        num_tests = num_tests or len(dataset)
        total = min(num_tests, len(dataset))
        passed = 0

        if status_box:
            status_box.info(f"Evaluating {total} samples with model `{self._agent.get_model()}`...")

        for i in range(total):
            sample = dataset[i]
            prompt = sample["prompt"]
            test_code = sample["test"]

            llm_output = self._agent.invoke(prompt)
            llm_code = self._code_cleaner.get_code_from_llm_output(llm_output)
            code_to_run = self._generate_test_code(llm_code, test_code)

            result = run_in_sandbox(code_to_run)
            if "Success" in result:
                passed += 1

            # Live progress update
            percent = int((i + 1) / total * 100)
            if progress_bar:
                progress_bar.progress(percent)
            if progress_text:
                progress_text.text(f"Evaluating: {percent}% ({i + 1}/{total})")

        output = EvalResult(
            passed=passed,
            total=total,
            score_in_percentage=f"{passed / total:.1%}",
            score_pass1=passed / total,
            model_name=self._agent.get_model()
        )

        if status_box:
            status_box.success("✅ Evaluation completed!")

        return output