import copy
import torch


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay

        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()

        for k in ema_state.keys():
            if ema_state[k].dtype.is_floating_point:
                ema_state[k].mul_(self.decay).add_(model_state[k], alpha=1 - self.decay)
            else:
                ema_state[k].copy_(model_state[k])
