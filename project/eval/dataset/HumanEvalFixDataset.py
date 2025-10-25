from datasets import load_dataset, Dataset
from pathlib import Path
from tqdm import tqdm
import pickle
import re


class HumanEvalFixDataset:
    def __init__(self):
        # Save path: 4 levels above the current file
        self.save_path = Path(__file__).resolve().parent.parent.parent.parent / "humanevalfix_cleaned.pkl"

        if self.save_path.exists():
            print(f"Loading cleaned dataset from {self.save_path}...")
            self.dataset = self._load_saved_dataset()
        else:
            print("Loading raw dataset from Hugging Face...")
            self.dataset = load_dataset("bigcode/humanevalpack", "python")['test']
            print("Cleaning dataset...")
            self.dataset = self._clean_dataset(self.dataset)
            print(f"Saving cleaned dataset to {self.save_path}...")
            self._save_dataset(self.dataset)

    def _clean_dataset(self, dataset: Dataset) -> Dataset:
        """
        Placeholder for cleaning logic.
        For example, you can:
            - strip whitespace
            - filter incomplete entries
            - standardize prompts or solutions
        """
        cleaned_data = []
        for item in tqdm(dataset, desc="Cleaning dataset"):
            cleaned_data.append({
                "prompt": item.get("declaration") + item.get("buggy_solution"),
                "test": item.get("test"),
            })
        return Dataset.from_list(cleaned_data)

    def _save_dataset(self, dataset: Dataset):
        """Save the cleaned dataset as a pickle file."""
        with open(self.save_path, "wb") as f:
            pickle.dump(dataset, f)

    def _load_saved_dataset(self) -> Dataset:
        """Load the cleaned dataset from pickle."""
        with open(self.save_path, "rb") as f:
            return pickle.load(f)
