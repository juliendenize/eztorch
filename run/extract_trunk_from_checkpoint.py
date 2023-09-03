from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch

if __name__ == "__main__":

    parser = ArgumentParser(
        description="Extract trunk from a checkpoint.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        type=str,
        help="Path to the raw checkpoint.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        type=str,
        help="Path to the output checkpoint.",
    )
    parser.add_argument(
        "--trunk-pattern",
        default="trunk.",
        type=str,
        help="Key to the trunk in state dict.",
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    out_checkpoint_path = args.output_path
    trunk_pattern = args.trunk_pattern

    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]

    old_keys = list(state_dict.keys())
    for old_key in old_keys:
        weight = state_dict.pop(old_key)
        if old_key.startswith(trunk_pattern):
            state_dict[old_key[len(trunk_pattern) :]] = weight

    torch.save(state_dict, out_checkpoint_path)
