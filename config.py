from dataclasses import dataclass


@dataclass
class Config:
    model_name: str = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
    dataset_name: str = "SargeDev/jev-distill-corpus-v3"
    max_choices: int = 255
