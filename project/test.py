from project.eval.evaluator import BugFixEvaluator


if __name__ == '__main__':

    evaluator = BugFixEvaluator()
    print(evaluator.evaluate(10))