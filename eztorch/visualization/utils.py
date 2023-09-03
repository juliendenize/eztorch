from typing import Iterable, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from matplotlib import rc
from torch import Tensor

rc("animation", html="jshtml")


def show_images(
    imgs: Union[Iterable[Tensor], Tensor], figsize: Iterable[float] = [6.4, 4.8]
):
    """Show images from a tensor or a list of tensor.

    Args:
        imgs: Images to display.
        figsize: Figure size for the images.
    """

    if not isinstance(imgs, list):
        imgs = [imgs]
    _, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=figsize)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = np.asarray(F.to_pil_image(img))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def show_video(video: Tensor) -> animation.ArtistAnimation:
    """Show a video thanks to animation from matplotlib.

    Args:
        video: The raw video to display.

    Returns:
        The animation to show.
    """
    video = video.long()
    video = np.asarray(video.permute(1, 2, 3, 0))

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    frames = [[ax.imshow(video[i], animated=True)] for i in range(len(video))]
    anim = animation.ArtistAnimation(fig, frames)
    plt.axis("off")
    plt.show()
    return anim
