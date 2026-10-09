from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.activations import ACT2FN
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    apply_mask_to_padding_states,
    causal_conv1d_fn,
    l2norm,
    torch_chunk_gated_delta_rule,
)

from config import Config

config = Config()


class ChoiceMask(NamedTuple):
    padding: torch.Tensor
    num_choices: int


class ChoiceGatedDeltaNet(Qwen3_5GatedDeltaNet):
    def forward(
        self,
        hidden_states,
        cache_params=None,
        attention_mask: ChoiceMask | None = None,
        **kwargs,
    ):
        hidden_states = apply_mask_to_padding_states(
            hidden_states, attention_mask.padding
        )
        batch_size, total_len, _ = hidden_states.shape
        seq_len = total_len - attention_mask.num_choices

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(
            batch_size, total_len, -1, self.head_v_dim
        )
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        weight = self.conv1d.weight.squeeze(1)
        prompt_qkv = mixed_qkv[:, :, :seq_len]
        prompt_conv = causal_conv1d_fn(
            prompt_qkv, weight, self.conv1d.bias, activation=self.activation
        )

        window = F.pad(prompt_qkv, (self.conv_kernel_size - 1, 0))[
            :, :, -(self.conv_kernel_size - 1) :
        ]
        choice_conv = (window * weight[:, :-1]).sum(-1, keepdim=True) + mixed_qkv[
            :, :, seq_len:
        ] * weight[:, -1:]
        if self.conv1d.bias is not None:
            choice_conv = choice_conv + self.conv1d.bias[:, None]
        choice_conv = ACT2FN[self.activation](choice_conv).to(prompt_conv.dtype)
        mixed_qkv = torch.cat([prompt_conv, choice_conv], dim=2).transpose(1, 2)

        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch_size, total_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, total_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, total_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        prompt_out, state = torch_chunk_gated_delta_rule(
            query[:, :seq_len],
            key[:, :seq_len],
            value[:, :seq_len],
            g=g[:, :seq_len],
            beta=beta[:, :seq_len],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

        state = state.float()
        choice_q = l2norm(query[:, seq_len:].float()) * self.head_k_dim**-0.5
        choice_k = l2norm(key[:, seq_len:].float())
        choice_v = value[:, seq_len:].float()
        choice_decay = g[:, seq_len:].exp()[..., None]
        choice_beta = beta[:, seq_len:].float()[..., None]
        read_q = torch.einsum("bhkv,bnhk->bnhv", state, choice_q)
        read_k = torch.einsum("bhkv,bnhk->bnhv", state, choice_k)
        delta = choice_beta * (choice_v - choice_decay * read_k)
        choice_out = (
            choice_decay * read_q + (choice_q * choice_k).sum(-1, keepdim=True) * delta
        )
        core_attn_out = torch.cat([prompt_out, choice_out.to(prompt_out.dtype)], dim=1)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z).reshape(batch_size, total_len, -1)
        return self.out_proj(core_attn_out)


class ChoiceModel(nn.Module):
    def __init__(
        self, model_name: str = config.model_name, dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.qwen = AutoModel.from_pretrained(
            model_name, dtype=dtype, attn_implementation="sdpa"
        ).language_model
        for module in self.qwen.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                module.__class__ = ChoiceGatedDeltaNet
        d_model = self.qwen.config.hidden_size

        choice_ids = self.tokenizer.encode(" choice", add_special_tokens=False)
        assert (
            len(choice_ids) == 1
        ), f'" choice" should be a single token, got {choice_ids}'
        choice_embedding = self.qwen.get_input_embeddings().weight[choice_ids[0]]
        self.initial_embedding = nn.Parameter(choice_embedding.detach().clone().float())
        self.position_embedding = nn.Embedding(config.max_choices, d_model)
        nn.init.normal_(
            self.position_embedding.weight, std=self.qwen.config.initializer_range
        )
        self.head = nn.Linear(d_model, 1)

    def forward(
        self,
        input_ids: torch.LongTensor,
        num_choices: int,
        attention_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if not 1 <= num_choices <= config.max_choices:
            raise ValueError(
                f"num_choices must be between 1 and {config.max_choices}, got {num_choices}"
            )
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        attention_mask = attention_mask.long()
        order = torch.argsort(attention_mask, dim=1, stable=True)
        input_ids = input_ids.gather(1, order)
        attention_mask = attention_mask.gather(1, order)

        token_embeds = self.qwen.get_input_embeddings()(input_ids)
        choice_embeds = self.initial_embedding + self.position_embedding(
            torch.arange(num_choices, device=device)
        )
        choice_embeds = choice_embeds.to(token_embeds.dtype).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([token_embeds, choice_embeds], dim=1)

        text_positions = (attention_mask.cumsum(1) - 1).clamp(min=0)
        choice_positions = attention_mask.sum(1, keepdim=True).expand(-1, num_choices)
        position_ids = torch.cat([text_positions, choice_positions], dim=1)

        total_len = seq_len + num_choices
        is_choice = torch.arange(total_len, device=device) >= seq_len
        key_valid = torch.cat(
            [attention_mask, attention_mask.new_ones(batch_size, num_choices)], dim=1
        ).bool()
        causal = torch.ones(
            total_len, total_len, dtype=torch.bool, device=device
        ).tril()
        allowed = (causal | is_choice[None, :])[None] & key_valid[:, None, :]
        dtype = inputs_embeds.dtype
        full_attention_mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
        full_attention_mask = full_attention_mask.masked_fill(
            ~allowed, torch.finfo(dtype).min
        )[:, None]

        hidden_states = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask={
                "full_attention": full_attention_mask,
                "linear_attention": ChoiceMask(key_valid, num_choices),
            },
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state

        choice_states = hidden_states[:, seq_len:].to(self.head.weight.dtype)
        scores = self.head(choice_states).squeeze(-1)
        return list(scores.unbind(dim=-1))
