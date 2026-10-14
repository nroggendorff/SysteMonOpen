from bisect import bisect_right

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from config import Config

config = Config()
tokenizer = AutoTokenizer.from_pretrained(config.model_name)


def format_prompt(
    state: str, question: str, options: list[str]
) -> tuple[str, list[int]]:
    text = "\n".join([state.strip(), "", question.strip(), "", "Options:"])
    option_ends = []
    for i, option in enumerate(options):
        text += f"\n{i + 1}. {option}"
        option_ends.append(len(text))
    return text, option_ends


def load_choice_dataset(split: str = "train", num_proc: int | None = None) -> Dataset:
    dataset = load_dataset(config.dataset_name, split=split)

    def is_valid(example) -> bool:
        options, target = example["options"], example["target"]
        if not 1 <= len(options) <= config.max_choices or len(target) != len(options):
            return False
        return min(target) >= 0 and sum(target) > 0

    def convert(example) -> dict:
        text, option_ends = format_prompt(
            example["state"], example["question"], example["options"]
        )
        encoding = tokenizer(text, return_offsets_mapping=True)
        token_starts = [start for start, _ in encoding.offset_mapping]
        total = sum(example["target"])
        return {
            "input_ids": encoding.input_ids,
            "choice_positions": [
                bisect_right(token_starts, end - 1) - 1 for end in option_ends
            ],
            "target": [p / total for p in example["target"]],
        }

    dataset = dataset.filter(is_valid, num_proc=num_proc)
    return dataset.map(convert, remove_columns=dataset.column_names, num_proc=num_proc)
