from datasets import Dataset, load_dataset

from config import Config

config = Config()


def format_prompt(state: str, question: str, options: list[str]) -> str:
    lines = [state.strip(), "", question.strip(), "", "Options:"]
    lines += [f"{i + 1}. {option}" for i, option in enumerate(options)]
    return "\n".join(lines)


def load_choice_dataset(split: str = "train", num_proc: int | None = None) -> Dataset:
    dataset = load_dataset(config.dataset_name, split=split)

    def is_valid(example) -> bool:
        options, target = example["options"], example["target"]
        if not 1 <= len(options) <= config.max_choices or len(target) != len(options):
            return False
        return min(target) >= 0 and sum(target) > 0

    def convert(example) -> dict:
        total = sum(example["target"])
        return {
            "text": format_prompt(
                example["state"], example["question"], example["options"]
            ),
            "num_choices": len(example["options"]),
            "target": [p / total for p in example["target"]],
        }

    dataset = dataset.filter(is_valid, num_proc=num_proc)
    return dataset.map(convert, remove_columns=dataset.column_names, num_proc=num_proc)
