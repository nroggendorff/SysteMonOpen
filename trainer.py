import math

import torch
import torch.nn.functional as F
from datasets import Dataset

from model import ChoiceModel


class Trainer:
    def __init__(
        self,
        model: ChoiceModel,
        lr: float = 1e-4,
        grad_accum_steps: int = 16,
        log_every: int = 10,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.device = device
        self.grad_accum_steps = grad_accum_steps
        self.log_every = log_every
        self.optimizer = torch.optim.AdamW([model.axes], lr=lr, fused=True)

    def score(self, example) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = torch.tensor([example["input_ids"]], device=self.device)
        choice_positions = torch.tensor(
            [example["choice_positions"]], device=self.device
        )
        scores = self.model(input_ids, choice_positions)[:, 0]
        target = torch.tensor(example["target"], device=self.device)
        loss = F.cross_entropy(scores, target.expand_as(scores), reduction="none")
        top1 = target[scores.argmax(-1)]
        return loss, top1

    def train(self, dataset: Dataset, epochs: int = 1, seed: int = 0):
        self.optimizer.zero_grad()
        step = 0
        losses, top1s, uniform = [], [], []
        for epoch in range(epochs):
            shuffled = dataset.shuffle(seed=seed + epoch)

            for i, example in enumerate(shuffled):
                loss, top1 = self.score(example)
                (loss.sum() / self.grad_accum_steps).backward()
                losses.append(loss.detach())
                top1s.append(top1)
                uniform.append(math.log(len(example["target"])))

                if (i + 1) % self.grad_accum_steps == 0 or i + 1 == len(shuffled):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    step += 1
                    if step % self.log_every == 0:
                        mean_loss = torch.stack(losses).mean(0)
                        mean_top1 = torch.stack(top1s).mean(0)
                        layer = mean_loss.argmin().item()
                        print(
                            f"epoch {epoch} step {step} uniform {sum(uniform) / len(uniform):.4f} "
                            f"best layer {layer} loss {mean_loss[layer]:.4f} top1 {mean_top1[layer]:.3f}"
                        )
                        losses.clear()
                        top1s.clear()
                        uniform.clear()

    @torch.no_grad()
    def evaluate(self, dataset: Dataset) -> dict:
        losses, top1s, num_choices = [], [], []
        for example in dataset:
            loss, top1 = self.score(example)
            losses.append(loss)
            top1s.append(top1)
            num_choices.append(len(example["target"]))
        return {
            "loss": torch.stack(losses).mean(0).cpu(),
            "top1": torch.stack(top1s).mean(0).cpu(),
            "uniform_loss": sum(map(math.log, num_choices)) / len(num_choices),
            "uniform_top1": sum(1 / n for n in num_choices) / len(num_choices),
        }
