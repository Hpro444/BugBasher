from tqdm import tqdm
import re
from project.sandbox import run_in_sandbox
from project.agent.model import BugBashAgent
from project.eval.dataset import HumanEvalFixDataset
from typing import Optional
import textwrap


class BugFixEvaluator:

    def __init__(self, agent: Optional[BugBashAgent] = None, dataset: Optional[HumanEvalFixDataset] = None):
        self._agent = agent if agent is not None else BugBashAgent()
        self._dataset = dataset if dataset is not None else HumanEvalFixDataset()

    @staticmethod
    def _remove_thinking_tags(llm_output: str) -> str:
        """
        Removes <think>...</think> sections from the LLM output.
        Returns the cleaned string even if the tags are missing or incomplete.
        """
        if not isinstance(llm_output, str):
            return ""

        if "</think>" in llm_output:
            parts = llm_output.split("</think>", 1)
            return parts[1].strip()

        if "<think>" in llm_output:
            return llm_output.split("<think>", 1)[0].strip()

        return llm_output.strip()

    @staticmethod
    def _generate_test_code(llm_code: str, test_code: str) -> str:
        code = f"""
    {llm_code}

    {test_code}

    """
        # Dedent to remove unwanted leading spaces
        return textwrap.dedent(code).strip() + "\n"

    @staticmethod
    def _extract_code(llm_output: str) -> str:
        match = re.search(r"```(?:python)?\s*(.*?)\s*```", llm_output, re.DOTALL)
        if match:
            return match.group(1).strip()
        return llm_output.strip()

    def _get_code_from_llm_output(self, llm_output: str) -> str:
        llm_output = self._remove_thinking_tags(llm_output)
        llm_code = self._extract_code(llm_output)
        return llm_code

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
            self._agent.reset_state()
            llm_code = self._get_code_from_llm_output(llm_output)

            # Build the complete test code
            code_to_run = self._generate_test_code(llm_code, test_code)

            # Run inside sandbox
            result = run_in_sandbox(code_to_run)
            print(code_to_run)
            print(result)
            # Simple pass/fail detection
            if "Success" in result:
                passed += 1

        print(f"\n✅ Passed {passed}/{total} tests ({passed / total:.1%})")
        return passed, total
