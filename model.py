import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from config import Config

config = Config()


class ChoiceModel(nn.Module):
    def __init__(
        self, model_name: str = config.model_name, dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.qwen = AutoModel.from_pretrained(
            model_name, dtype=dtype, attn_implementation="sdpa"
        ).language_model
        self.qwen.requires_grad_(False)
        num_states = self.qwen.config.num_hidden_layers + 1
        self.axes = nn.Parameter(torch.zeros(num_states, self.qwen.config.hidden_size))

    def forward(
        self,
        input_ids: torch.LongTensor,
        choice_positions: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            hidden_states = self.qwen(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            ).hidden_states
            batch = torch.arange(input_ids.shape[0], device=input_ids.device)[:, None]
            choice_states = torch.stack(
                [h[batch, choice_positions] for h in hidden_states]
            ).float()

            choice_states = F.rms_norm(choice_states, choice_states.shape[-1:])
        return torch.einsum("lbnd,ld->lbn", choice_states, self.axes)
