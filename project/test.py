from project.eval.evaluator import BugFixEvaluator
from project.eval.dataset import HumanEvalFixDataset
from project.agent.model import BugBashAgent

if __name__ == "__main__":
    evaluator = BugFixEvaluator()
    evaluator.evaluate(1)

    # dataset = HumanEvalFixDataset()
    # print(dataset.dataset[0])

    # agent = BugBashAgent()
    #
    # while True:
    #     print(agent.invoke(input('> ')))
