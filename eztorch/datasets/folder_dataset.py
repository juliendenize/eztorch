# ------------------------------------------------------------------------
# Modified from Torchvision (https://github.com/pytorch/vision)
# Licensed under the BSD 3-Clause License
# -----------------------------

import os
import os.path
from typing import (Any, Callable, Dict, Iterable, List, Optional, Tuple,
                    Union, cast)

import torch
from lightning.pytorch.utilities import rank_zero_warn
from PIL import Image
from torchvision.datasets.vision import VisionDataset


def has_file_allowed_extension(
    filename: str, extensions: Union[str, Tuple[str, ...]]
) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename: Path to a file.
        extensions: Extensions to consider (lowercase).

    Returns:
        ``True`` if the filename ends with one of given extensions.
    """
    return filename.lower().endswith(
        extensions if isinstance(extensions, str) else tuple(extensions)
    )


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename: Path to a file.

    Returns:
        ``True`` if the filename ends with a known image extension.
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The ``class_to_idx`` parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError(
            "'class_to_index' must have at least one entry to collect any samples."
        )

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time"
        )

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            # type: ignore[arg-type]
            return has_file_allowed_extension(x, extensions)

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = (
            f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        )
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root: Root directory path.
        loader: A function to load a sample given its path.
        extensions: A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
        is_valid_file: A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
        class_ratio: Ratio of classes to use if ``class_list`` is ``None``.
        sample_ratio: Ratio of samples to use.
        class_list: If not ``None``, list of classes to use.
        sample_list_path: If not ``None``, list of samples to use.
        seed: If not ``None``, seed used to randomly choose class and samples.
    """

    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        class_ratio: float = 1.0,
        sample_ratio: float = 1.0,
        class_list: Optional[Iterable[str]] = None,
        sample_list_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        assert (
            0 < class_ratio <= 1.0
        ), "class_ratio should be comprised between 0 (excluded) and 1. (included)"
        assert (
            0 < sample_ratio <= 1.0
        ), "sample_ratio should be comprised between 0 (excluded) and 1. (included)"

        classes, class_to_idx = self.find_classes(self.root)

        if class_ratio < 1.0 or sample_ratio < 1.0:
            global_seed = int(os.getenv("PL_GLOBAL_SEED"))
            if global_seed is None and seed is None:
                rank_zero_warn(
                    "PL_GLOBAL_SEED environment variable is not defined as well as the seed argument, the default seed used is 0 for class_ratio and sample_ratio."
                )
            seed = seed or global_seed or 0
            g = torch.Generator()
            g.manual_seed(seed)

        if class_list is not None:
            for cls in class_list:
                if cls not in classes:
                    raise AttributeError(
                        f"Class {cls} specified in class_list not found in folder."
                    )
            classes = sorted(class_list)
            class_to_idx = {class_name: i for i, class_name in enumerate(classes)}

        elif class_ratio < 1.0:
            num_classes = round(len(classes) * class_ratio)
            indices = list(torch.randperm(len(classes))[:num_classes])

            classes = sorted([classes[idx] for idx in indices])
            class_to_idx = {cls: class_to_idx[cls] for cls in classes}

        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        if sample_list_path is not None:
            with open(sample_list_path) as f:
                sample_list = f.read()
            sample_list = sample_list.split("\n")
            fn_to_sample = {elt[0].split("/")[-1]: elt for elt in samples}
            samples = [fn_to_sample[file] for file in sample_list]

        if sample_ratio < 1.0:
            num_images = round(len(samples) * sample_ratio)
            indices = list(torch.randperm(len(samples))[:num_images])
            samples = [samples[idx] for idx in indices]

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory: Root dataset directory, corresponding to ``self.root``.
            class_to_idx: Dictionary mapping class name to class index.
            extensions: A list of allowed extensions.
                Either extensions or is_valid_file should be passed.
            is_valid_file: A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                ``is_valid_file`` should not be passed.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are ``None`` or both are not ``None``.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            Samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(
            directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file
        )

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory: Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            Where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        sample_dict = {"input": sample, "label": target, "idx": index}

        return sample_dict

    def __len__(self) -> int:
        return len(self.samples)


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root: Root directory path.
        transform: A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader: A function to load an image given its path.
        is_valid_file: A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        class_ratio: Ratio of classes to use if ``class_list`` is ``None``. Defaults to :math:`1`.
        sample_ratio: Ratio of samples to use. Defaults to `math`:1:.
        class_list: If not ``None``, List of classes to use. Defaults to ``None``.
        seed: If not ``None``, seed used to randomly choose class and samples. Defaults to ``None``.

     Attributes:
        classes: List of the class names sorted alphabetically.
        class_to_idx: Dict with items (class_name, class_index).
        imgs: List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        class_ratio: float = 1.0,
        sample_ratio: float = 1.0,
        sample_list_path: Optional[str] = None,
        class_list: Optional[Iterable[str]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            class_ratio=class_ratio,
            sample_ratio=sample_ratio,
            sample_list_path=sample_list_path,
            class_list=class_list,
            seed=seed,
        )
        self.imgs = self.samples
