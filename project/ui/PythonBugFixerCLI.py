import argparse
import sys
from pathlib import Path


class PythonBugFixerCLI:
    def __init__(self):
        # Delay all heavy imports
        from project.config import Configuration
        from project.cleaners import LLOutputCleaner

        self.cleaner = LLOutputCleaner()
        self.config = Configuration()

    def _create_agent(self, model_name=None, url=None):
        """
        Create a BugBashAgent.
        If model_name or url is None, defaults from config will be used.
        """
        from project.agent.model import BugBashAgent

        if model_name or url:
            return BugBashAgent(model=model_name, url=url)
        return BugBashAgent()  # uses defaults from Configuration internally

    def fix(self, file_path: str, model_name: str = None, url: str = None):
        """Fix a Python file using the specified model and URL."""
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            sys.exit(1)

        with open(file_path, "r", encoding="utf-8") as f:
            buggy_code = f.read()

        print(f"🔧 Fixing code using model='{model_name or self.config.ollama_model}', "
              f"url='{url or self.config.ollama_url}'...\n")

        agent = self._create_agent(model_name, url)
        fixed_code = agent.invoke(buggy_code)
        fixed_code = self.cleaner.get_code_from_llm_output(fixed_code)

        output_path = file_path.with_name(f"{file_path.stem}_fixed.py")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(fixed_code)

        print(f"\n✅ Fixed code saved to: {output_path}")

    def evaluate(self, model_name: str = None, url: str = None, limit: int = None):
        """Evaluate a model using HumanEvalFixDataset."""
        # Lazy imports for eval mode only
        from project.eval.evaluator import BugFixEvaluator
        from project.eval.dataset import HumanEvalFixDataset

        print(f"📊 Evaluating model='{model_name or self.config.ollama_model}', "
              f"url='{url or self.config.ollama_url}'...\n")


        agent = self._create_agent(model_name, url)
        evaluator = BugFixEvaluator(agent)

        results = evaluator.evaluate(num_tests=limit)
        print("\n--- Evaluation Complete ---")
        print(results)

    def run(self):
        parser = argparse.ArgumentParser(description="🐞 Python Bug Fixer CLI")
        subparsers = parser.add_subparsers(dest="command")

        # Fix command
        fix_parser = subparsers.add_parser("fix", help="Fix a Python file")
        fix_parser.add_argument("file", help="Path to the Python file to fix")
        fix_parser.add_argument("--model", help="Model name to use (overrides config)")
        fix_parser.add_argument("--url", help="Ollama / API URL (overrides config)")

        # Evaluate command
        eval_parser = subparsers.add_parser("evaluate", help="Evaluate model on dataset")
        eval_parser.add_argument("--model", help="Model name to use (overrides config)")
        eval_parser.add_argument("--url", help="Ollama / API URL (overrides config)")
        eval_parser.add_argument(
            "--limit",
            type=int,
            help="Limit the number of evaluation tasks (e.g., 10 to test only first 10 items)",
        )

        args = parser.parse_args()

        # If no command or help requested, don't import anything heavy
        if not args.command:
            parser.print_help()
            sys.exit(0)

        if args.command == "fix":
            self.fix(args.file, args.model, args.url)
        elif args.command == "evaluate":
            self.evaluate(args.model, args.url, args.limit)
        else:
            parser.print_help()


