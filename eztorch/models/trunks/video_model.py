import hydra
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Module


class VideoHeadModel(Module):
    """A general purpose model that handles an encoder and its head.

    Args:
        model: A model that precedes the head and is supposed to have initialized weights. Ex: stem + stages.
        head: A network head.
    """

    def __init__(self, model: Module, head: Module):
        super().__init__()
        self.model = model
        self.head = head

    def forward(self, x: Tensor):
        x = self.model(x)
        x = self.head(x)

        return x


def create_video_head_model(
    model: DictConfig,
    head: DictConfig,
):
    """Build a video model.

    Args:
        model: Config for the model.
        head: Config for the head.
    """

    model = hydra.utils.instantiate(model)
    head = hydra.utils.instantiate(head)

    video_model = VideoHeadModel(model, head)

    return video_model
