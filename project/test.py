from project.eval.evaluator import BugFixEvaluator
from project.eval.dataset import HumanEvalFixDataset
from project.agent.model import BugBashAgent
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Adds two numbers together"""
    return a + b


if __name__ == "__main__":
    evaluator = BugFixEvaluator()
    evaluator.evaluate(1)

    # dataset = HumanEvalFixDataset()
    # print(dataset.dataset[0])

    # agent = BugBashAgent(tools=[add])
    #
    # while True:
    #     print(agent.invoke(input('> ')))
