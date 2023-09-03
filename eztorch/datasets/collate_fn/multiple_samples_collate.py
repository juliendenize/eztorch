from typing import Any, Dict, List

from torch.utils.data._utils.collate import default_collate


def multiple_samples_collate(batch: List[Dict[str, List[Any]]]) -> Dict[str, Any]:
    """Collate function for repeated augmentation. Each instance in the batch has more than one sample.

    Args:
        batch: Batch of data before collate.

    Returns:
        The collated batch.
    """

    batch_dict = {}
    if type(batch[0]) is not dict:
        keys = batch[0][0].keys()
    else:
        keys = batch[0].keys()
    for k in keys:
        v_iter = []
        for samples_dict in batch:
            if type(samples_dict) is dict:
                samples_dict = [samples_dict]
            for sample_dict in samples_dict:
                v_iter += [sample_dict[k]]
        batch_dict[k] = default_collate(v_iter)

    return batch_dict
