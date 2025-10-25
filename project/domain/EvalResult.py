from dataclasses import dataclass

@dataclass
class EvalResult:
    passed: int
    total: int
    model_name: str
    score_in_percentage: str
    score_pass1: float

    def __str__(self):
        return (f"Evaluation Result for {self.model_name}:\n"
                f"  Passed: {self.passed}/{self.total}\n"
                f"  Score: {self.score_in_percentage}\n"
                f"  Score Pass@1: {self.score_pass1:.2f}")
