"""Exponential Moving Average of model parameters."""

import torch


class EMAModel:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.target_decay = decay
        self.step = 0
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        self.step += 1
        decay = min(self.target_decay, (1 + self.step) / (10 + self.step))
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(decay).add_(p.data, alpha=1 - decay)

    def apply(self, model: torch.nn.Module):
        """Swap model params with EMA shadow params. Call again to restore."""
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                p.data, self.shadow[name] = self.shadow[name], p.data.clone()

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.clone() for k, v in state_dict.items()}
