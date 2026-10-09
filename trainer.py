import torch
import torch.nn.functional as F
from datasets import Dataset

from model import ChoiceModel


class Trainer:
    def __init__(
        self,
        model: ChoiceModel,
        lr: float = 1e-5,
        grad_accum_steps: int = 16,
        max_grad_norm: float = 1.0,
        gradient_checkpointing: bool = True,
        log_every: int = 10,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.device = device
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.log_every = log_every
        if gradient_checkpointing:
            self.model.qwen.gradient_checkpointing_enable()
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )

    def train(self, dataset: Dataset, epochs: int = 1, seed: int = 0):
        self.model.train()
        self.optimizer.zero_grad()
        step = 0
        losses = []
        for epoch in range(epochs):
            shuffled = dataset.shuffle(seed=seed + epoch)

            for i, example in enumerate(shuffled):
                input_ids = self.model.tokenizer(
                    example["text"], return_tensors="pt"
                ).input_ids.to(self.device)
                scores = torch.stack(
                    self.model(input_ids, example["num_choices"]), dim=-1
                )
                target = torch.tensor(
                    [example["target"]], dtype=scores.dtype, device=self.device
                )
                loss = F.cross_entropy(scores, target)
                (loss / self.grad_accum_steps).backward()
                losses.append(loss.item())

                if (i + 1) % self.grad_accum_steps == 0 or i + 1 == len(shuffled):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    step += 1
                    if step % self.log_every == 0:
                        print(
                            f"epoch {epoch} step {step} loss {sum(losses) / len(losses):.4f}"
                        )
                        losses.clear()
