import unittest

import torch

from eztorch.losses.sce_token_loss import (compute_sce_token_loss,
                                           compute_sce_token_masks)


class TestSCETokenMasks(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 2
        self.num_tokens = 8
        self.num_negatives = 2

    def test_one_device_zero_pos_radius_no_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=0,
            keep_aligned_positive=True,
            use_keys=False,
            use_all_keys=False,
            rank=0,
            world_size=1,
        )

        expected_mask_prob_q = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ],
            device=mask_prob_q.device,
            dtype=mask_prob_q.dtype,
        )

        expected_mask_sim_q = torch.tensor(
            [
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            ],
            device=mask_sim_q.device,
            dtype=mask_sim_q.dtype,
        )

        expected_mask_sim_k = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            device=mask_sim_k.device,
            dtype=mask_sim_k.dtype,
        )

        expected_num_positives = torch.tensor(
            [
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
            ],
            device=num_positives_per_token.device,
            dtype=num_positives_per_token.dtype,
        )

        expected_mask_log_q = torch.cat(
            (
                (1 - mask_sim_q),
                torch.ones(
                    (mask_sim_q.shape[0], self.num_negatives),
                    device=mask_sim_q.device,
                    dtype=mask_sim_q.dtype,
                ),
            ),
            1,
        ).to(dtype=torch.bool)

        assert torch.allclose(mask_prob_q, expected_mask_prob_q)
        assert torch.allclose(mask_sim_k, expected_mask_sim_k)
        assert torch.allclose(mask_sim_q, expected_mask_sim_q)
        assert torch.allclose(mask_log_q, expected_mask_log_q)
        assert torch.allclose(num_positives_per_token, expected_num_positives)

    def test_one_device_two_pos_radius_no_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=True,
            use_keys=False,
            use_all_keys=False,
            rank=0,
            world_size=1,
        )

        expected_mask_prob_q = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            ],
            device=mask_prob_q.device,
            dtype=mask_prob_q.dtype,
        )

        expected_mask_sim_q = torch.tensor(
            [
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            ],
            device=mask_sim_q.device,
            dtype=mask_sim_q.dtype,
        )

        expected_mask_sim_k = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            device=mask_sim_k.device,
            dtype=mask_sim_k.dtype,
        )

        expected_num_positives = torch.tensor(
            [
                [3],
                [4],
                [5],
                [5],
                [5],
                [5],
                [4],
                [3],
                [3],
                [4],
                [5],
                [5],
                [5],
                [5],
                [4],
                [3],
            ],
            device=num_positives_per_token.device,
            dtype=num_positives_per_token.dtype,
        )

        expected_mask_log_q = torch.cat(
            (
                (1 - mask_sim_q),
                torch.ones(
                    (mask_sim_q.shape[0], self.num_negatives),
                    device=mask_sim_q.device,
                    dtype=mask_sim_q.dtype,
                ),
            ),
            1,
        ).to(dtype=torch.bool)

        assert torch.allclose(
            mask_prob_q, expected_mask_prob_q / expected_num_positives
        )
        assert torch.allclose(mask_sim_k, expected_mask_sim_k)
        assert torch.allclose(mask_sim_q, expected_mask_sim_q)
        assert torch.allclose(mask_log_q, expected_mask_log_q)
        assert torch.allclose(num_positives_per_token, expected_num_positives)

    def test_one_device_two_pos_radius_no_keys_not_aligned_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=False,
            use_keys=False,
            use_all_keys=False,
            rank=0,
            world_size=1,
        )

        expected_mask_prob_q = torch.tensor(
            [
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            ],
            device=mask_prob_q.device,
            dtype=mask_prob_q.dtype,
        )

        expected_mask_sim_q = torch.tensor(
            [
                [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
            ],
            device=mask_sim_q.device,
            dtype=mask_sim_q.dtype,
        )

        expected_mask_sim_k = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            device=mask_sim_k.device,
            dtype=mask_sim_k.dtype,
        )

        expected_num_positives = torch.tensor(
            [
                [2],
                [3],
                [4],
                [4],
                [4],
                [4],
                [3],
                [2],
                [2],
                [3],
                [4],
                [4],
                [4],
                [4],
                [3],
                [2],
            ],
            device=num_positives_per_token.device,
            dtype=num_positives_per_token.dtype,
        )

        expected_mask_log_q = torch.cat(
            (
                (1 - mask_sim_q),
                torch.ones(
                    (mask_sim_q.shape[0], self.num_negatives),
                    device=mask_sim_q.device,
                    dtype=mask_sim_q.dtype,
                ),
            ),
            1,
        ).to(dtype=torch.bool)

        assert torch.allclose(
            mask_prob_q, expected_mask_prob_q / expected_num_positives
        )
        assert torch.allclose(mask_sim_k, expected_mask_sim_k)
        assert torch.allclose(mask_sim_q, expected_mask_sim_q)
        assert torch.allclose(mask_log_q, expected_mask_log_q)
        assert torch.allclose(num_positives_per_token, expected_num_positives)

    def test_several_devices_zero_pos_radius_no_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=0,
            keep_aligned_positive=True,
            use_keys=False,
            use_all_keys=False,
            rank=1,
            world_size=3,
        )

        expected_mask_prob_q = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ],
            device=mask_prob_q.device,
            dtype=mask_prob_q.dtype,
        )

        expected_mask_sim_q = torch.tensor(
            [
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            ],
            device=mask_sim_q.device,
            dtype=mask_sim_q.dtype,
        )

        expected_mask_sim_k = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            device=mask_sim_k.device,
            dtype=mask_sim_k.dtype,
        )

        expected_num_positives = torch.tensor(
            [
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
            ],
            device=num_positives_per_token.device,
            dtype=num_positives_per_token.dtype,
        )

        expected_mask_log_q = torch.cat(
            (
                (1 - mask_sim_q),
                torch.ones(
                    (mask_sim_q.shape[0], self.num_negatives),
                    device=mask_sim_q.device,
                    dtype=mask_sim_q.dtype,
                ),
            ),
            1,
        ).to(dtype=torch.bool)

        assert torch.allclose(mask_prob_q, expected_mask_prob_q)
        assert torch.allclose(mask_sim_k, expected_mask_sim_k)
        assert torch.allclose(mask_sim_q, expected_mask_sim_q)
        assert torch.allclose(mask_log_q, expected_mask_log_q)
        assert torch.allclose(num_positives_per_token, expected_num_positives)

    def test_several_devices_two_pos_radius_no_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=True,
            use_keys=False,
            use_all_keys=False,
            rank=1,
            world_size=3,
        )

        expected_mask_prob_q = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            ],
            device=mask_prob_q.device,
            dtype=mask_prob_q.dtype,
        )

        expected_mask_sim_q = torch.tensor(
            [
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            ],
            device=mask_sim_q.device,
            dtype=mask_sim_q.dtype,
        )

        expected_mask_sim_k = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            device=mask_sim_k.device,
            dtype=mask_sim_k.dtype,
        )

        expected_num_positives = torch.tensor(
            [
                [3],
                [4],
                [5],
                [5],
                [5],
                [5],
                [4],
                [3],
                [3],
                [4],
                [5],
                [5],
                [5],
                [5],
                [4],
                [3],
            ],
            device=num_positives_per_token.device,
            dtype=num_positives_per_token.dtype,
        )

        expected_mask_log_q = torch.cat(
            (
                (1 - mask_sim_q),
                torch.ones(
                    (mask_sim_q.shape[0], self.num_negatives),
                    device=mask_sim_q.device,
                    dtype=mask_sim_q.dtype,
                ),
            ),
            1,
        ).to(dtype=torch.bool)

        assert torch.allclose(
            mask_prob_q, expected_mask_prob_q / expected_num_positives
        )
        assert torch.allclose(mask_sim_k, expected_mask_sim_k)
        assert torch.allclose(mask_sim_q, expected_mask_sim_q)
        assert torch.allclose(mask_log_q, expected_mask_log_q)
        assert torch.allclose(num_positives_per_token, expected_num_positives)

    def test_several_devices_two_pos_radius_no_keys_not_aligned_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=False,
            use_keys=False,
            use_all_keys=False,
            rank=1,
            world_size=3,
        )

        expected_mask_prob_q = torch.tensor(
            [
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            ],
            device=mask_prob_q.device,
            dtype=mask_prob_q.dtype,
        )

        expected_mask_sim_q = torch.tensor(
            [
                [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
            ],
            device=mask_sim_q.device,
            dtype=mask_sim_q.dtype,
        )

        expected_mask_sim_k = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            device=mask_sim_k.device,
            dtype=mask_sim_k.dtype,
        )

        expected_num_positives = torch.tensor(
            [
                [2],
                [3],
                [4],
                [4],
                [4],
                [4],
                [3],
                [2],
                [2],
                [3],
                [4],
                [4],
                [4],
                [4],
                [3],
                [2],
            ],
            device=num_positives_per_token.device,
            dtype=num_positives_per_token.dtype,
        )

        expected_mask_log_q = torch.cat(
            (
                (1 - mask_sim_q),
                torch.ones(
                    (mask_sim_q.shape[0], self.num_negatives),
                    device=mask_sim_q.device,
                    dtype=mask_sim_q.dtype,
                ),
            ),
            1,
        ).to(dtype=torch.bool)

        assert torch.allclose(
            mask_prob_q, expected_mask_prob_q / expected_num_positives
        )
        assert torch.allclose(mask_sim_k, expected_mask_sim_k)
        assert torch.allclose(mask_sim_q, expected_mask_sim_q)
        assert torch.allclose(mask_log_q, expected_mask_log_q)
        assert torch.allclose(num_positives_per_token, expected_num_positives)

    def test_one_device_zero_pos_radius_with_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=0,
            keep_aligned_positive=True,
            use_keys=True,
            use_all_keys=False,
            rank=0,
            world_size=1,
        )

        expected_mask_prob_q = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ],
            device=mask_prob_q.device,
            dtype=mask_prob_q.dtype,
        )

        expected_mask_sim_q = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            device=mask_sim_q.device,
            dtype=mask_sim_q.dtype,
        )

        expected_mask_sim_k = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            ],
            device=mask_sim_k.device,
            dtype=mask_sim_k.dtype,
        )

        expected_num_positives = torch.tensor(
            [
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
            ],
            device=num_positives_per_token.device,
            dtype=num_positives_per_token.dtype,
        )

        expected_mask_log_q = torch.cat(
            (
                (1 - mask_sim_q),
                torch.ones(
                    (mask_sim_q.shape[0], self.num_negatives),
                    device=mask_sim_q.device,
                    dtype=mask_sim_q.dtype,
                ),
            ),
            1,
        ).to(dtype=torch.bool)

        assert torch.allclose(mask_prob_q, expected_mask_prob_q)
        assert torch.allclose(mask_sim_k, expected_mask_sim_k)
        assert torch.allclose(mask_sim_q, expected_mask_sim_q)
        assert torch.allclose(mask_log_q, expected_mask_log_q)
        assert torch.allclose(num_positives_per_token, expected_num_positives)

    def test_one_device_two_pos_radius_with_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=True,
            use_keys=True,
            use_all_keys=False,
            rank=0,
            world_size=1,
        )

        expected_mask_prob_q = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            ],
            device=mask_prob_q.device,
            dtype=mask_prob_q.dtype,
        )

        expected_mask_sim_q = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            device=mask_sim_q.device,
            dtype=mask_sim_q.dtype,
        )

        expected_mask_sim_k = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
            ],
            device=mask_sim_k.device,
            dtype=mask_sim_k.dtype,
        )

        expected_num_positives = torch.tensor(
            [
                [3],
                [4],
                [5],
                [5],
                [5],
                [5],
                [4],
                [3],
                [3],
                [4],
                [5],
                [5],
                [5],
                [5],
                [4],
                [3],
            ],
            device=num_positives_per_token.device,
            dtype=num_positives_per_token.dtype,
        )

        expected_mask_log_q = torch.cat(
            (
                (1 - mask_sim_q),
                torch.ones(
                    (mask_sim_q.shape[0], self.num_negatives),
                    device=mask_sim_q.device,
                    dtype=mask_sim_q.dtype,
                ),
            ),
            1,
        ).to(dtype=torch.bool)

        assert torch.allclose(
            mask_prob_q, expected_mask_prob_q / expected_num_positives
        )
        assert torch.allclose(mask_sim_k, expected_mask_sim_k)
        assert torch.allclose(mask_sim_q, expected_mask_sim_q)
        assert torch.allclose(mask_log_q, expected_mask_log_q)
        assert torch.allclose(num_positives_per_token, expected_num_positives)

    def test_one_device_two_pos_radius_with_keys_not_aligned_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=False,
            use_keys=True,
            use_all_keys=False,
            rank=0,
            world_size=1,
        )

        expected_mask_prob_q = torch.tensor(
            [
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            ],
            device=mask_prob_q.device,
            dtype=mask_prob_q.dtype,
        )

        expected_mask_sim_q = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            ],
            device=mask_sim_q.device,
            dtype=mask_sim_q.dtype,
        )

        expected_mask_sim_k = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
            ],
            device=mask_sim_k.device,
            dtype=mask_sim_k.dtype,
        )

        expected_num_positives = torch.tensor(
            [
                [2],
                [3],
                [4],
                [4],
                [4],
                [4],
                [3],
                [2],
                [2],
                [3],
                [4],
                [4],
                [4],
                [4],
                [3],
                [2],
            ],
            device=num_positives_per_token.device,
            dtype=num_positives_per_token.dtype,
        )

        expected_mask_log_q = torch.cat(
            (
                (1 - mask_sim_q),
                torch.ones(
                    (mask_sim_q.shape[0], self.num_negatives),
                    device=mask_sim_q.device,
                    dtype=mask_sim_q.dtype,
                ),
            ),
            1,
        ).to(dtype=torch.bool)

        assert torch.allclose(
            mask_prob_q, expected_mask_prob_q / expected_num_positives
        )
        assert torch.allclose(mask_sim_k, expected_mask_sim_k)
        assert torch.allclose(mask_sim_q, expected_mask_sim_q)
        assert torch.allclose(mask_log_q, expected_mask_log_q)
        assert torch.allclose(num_positives_per_token, expected_num_positives)

    def test_several_devices_zero_pos_radius_with_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=0,
            keep_aligned_positive=True,
            use_keys=True,
            use_all_keys=False,
            rank=1,
            world_size=3,
        )

        expected_mask_prob_q = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ],
            device=mask_prob_q.device,
            dtype=mask_prob_q.dtype,
        )

        expected_mask_sim_q = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            device=mask_sim_q.device,
            dtype=mask_sim_q.dtype,
        )

        expected_mask_sim_k = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            ],
            device=mask_sim_k.device,
            dtype=mask_sim_k.dtype,
        )

        expected_num_positives = torch.tensor(
            [
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
            ],
            device=num_positives_per_token.device,
            dtype=num_positives_per_token.dtype,
        )

        expected_mask_log_q = torch.cat(
            (
                (1 - mask_sim_q),
                torch.ones(
                    (mask_sim_q.shape[0], self.num_negatives),
                    device=mask_sim_q.device,
                    dtype=mask_sim_q.dtype,
                ),
            ),
            1,
        ).to(dtype=torch.bool)

        assert torch.allclose(mask_prob_q, expected_mask_prob_q)
        assert torch.allclose(mask_sim_k, expected_mask_sim_k)
        assert torch.allclose(mask_sim_q, expected_mask_sim_q)
        assert torch.allclose(mask_log_q, expected_mask_log_q)
        assert torch.allclose(num_positives_per_token, expected_num_positives)

    def test_several_devices_two_pos_radius_with_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=True,
            use_keys=True,
            use_all_keys=False,
            rank=1,
            world_size=3,
        )

        expected_mask_prob_q = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            ],
            device=mask_prob_q.device,
            dtype=mask_prob_q.dtype,
        )

        expected_mask_sim_q = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            device=mask_sim_q.device,
            dtype=mask_sim_q.dtype,
        )

        expected_mask_sim_k = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
            ],
            device=mask_sim_k.device,
            dtype=mask_sim_k.dtype,
        )

        expected_num_positives = torch.tensor(
            [
                [3],
                [4],
                [5],
                [5],
                [5],
                [5],
                [4],
                [3],
                [3],
                [4],
                [5],
                [5],
                [5],
                [5],
                [4],
                [3],
            ],
            device=num_positives_per_token.device,
            dtype=num_positives_per_token.dtype,
        )

        expected_mask_log_q = torch.cat(
            (
                (1 - mask_sim_q),
                torch.ones(
                    (mask_sim_q.shape[0], self.num_negatives),
                    device=mask_sim_q.device,
                    dtype=mask_sim_q.dtype,
                ),
            ),
            1,
        ).to(dtype=torch.bool)

        assert torch.allclose(
            mask_prob_q, expected_mask_prob_q / expected_num_positives
        )
        assert torch.allclose(mask_sim_k, expected_mask_sim_k)
        assert torch.allclose(mask_sim_q, expected_mask_sim_q)
        assert torch.allclose(mask_log_q, expected_mask_log_q)
        assert torch.allclose(num_positives_per_token, expected_num_positives)

    def test_several_devices_two_pos_radius_with_keys_not_aligned_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=False,
            use_keys=True,
            use_all_keys=False,
            rank=1,
            world_size=3,
        )

        expected_mask_prob_q = torch.tensor(
            [
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            ],
            device=mask_prob_q.device,
            dtype=mask_prob_q.dtype,
        )

        expected_mask_sim_q = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            ],
            device=mask_sim_q.device,
            dtype=mask_sim_q.dtype,
        )

        expected_mask_sim_k = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
            ],
            device=mask_sim_k.device,
            dtype=mask_sim_k.dtype,
        )

        expected_num_positives = torch.tensor(
            [
                [2],
                [3],
                [4],
                [4],
                [4],
                [4],
                [3],
                [2],
                [2],
                [3],
                [4],
                [4],
                [4],
                [4],
                [3],
                [2],
            ],
            device=num_positives_per_token.device,
            dtype=num_positives_per_token.dtype,
        )

        expected_mask_log_q = torch.cat(
            (
                (1 - mask_sim_q),
                torch.ones(
                    (mask_sim_q.shape[0], self.num_negatives),
                    device=mask_sim_q.device,
                    dtype=mask_sim_q.dtype,
                ),
            ),
            1,
        ).to(dtype=torch.bool)

        assert torch.allclose(
            mask_prob_q, expected_mask_prob_q / expected_num_positives
        )
        assert torch.allclose(mask_sim_k, expected_mask_sim_k)
        assert torch.allclose(mask_sim_q, expected_mask_sim_q)
        assert torch.allclose(mask_log_q, expected_mask_log_q)
        assert torch.allclose(num_positives_per_token, expected_num_positives)

    def test_one_device_zero_pos_radius_with_all_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=0,
            keep_aligned_positive=True,
            use_keys=False,
            use_all_keys=True,
            rank=0,
            world_size=1,
        )

        expected_mask_prob_q = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ],
            device=mask_prob_q.device,
            dtype=mask_prob_q.dtype,
        )

        expected_mask_sim_k = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ],
            device=mask_sim_k.device,
            dtype=mask_sim_k.dtype,
        )

        expected_num_positives = torch.tensor(
            [
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
            ],
            device=num_positives_per_token.device,
            dtype=num_positives_per_token.dtype,
        )

        assert torch.allclose(mask_prob_q, expected_mask_prob_q)
        assert torch.allclose(mask_sim_k, expected_mask_sim_k)
        assert mask_sim_q is None
        assert mask_log_q is None
        assert torch.allclose(num_positives_per_token, expected_num_positives)

    def test_one_device_two_pos_radius_with_all_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=True,
            use_keys=False,
            use_all_keys=True,
            rank=0,
            world_size=1,
        )

        expected_mask_prob_q = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            ],
            device=mask_prob_q.device,
            dtype=mask_prob_q.dtype,
        )

        expected_mask_sim_k = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            ],
            device=mask_sim_k.device,
            dtype=mask_sim_k.dtype,
        )

        expected_num_positives = torch.tensor(
            [
                [3],
                [4],
                [5],
                [5],
                [5],
                [5],
                [4],
                [3],
                [3],
                [4],
                [5],
                [5],
                [5],
                [5],
                [4],
                [3],
            ],
            device=num_positives_per_token.device,
            dtype=num_positives_per_token.dtype,
        )

        assert torch.allclose(
            mask_prob_q, expected_mask_prob_q / expected_num_positives
        )
        assert torch.allclose(mask_sim_k, expected_mask_sim_k)
        assert mask_sim_q is None
        assert mask_log_q is None
        assert torch.allclose(num_positives_per_token, expected_num_positives)

    def test_one_device_two_pos_radius_with_all_keys_not_aligned_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=False,
            use_keys=False,
            use_all_keys=True,
            rank=0,
            world_size=1,
        )

        expected_mask_prob_q = torch.tensor(
            [
                [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            ],
            device=mask_prob_q.device,
            dtype=mask_prob_q.dtype,
        )

        expected_mask_sim_q = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ],
            device=mask_sim_q.device,
            dtype=mask_sim_q.dtype,
        )

        expected_mask_sim_k = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            ],
            device=mask_sim_k.device,
            dtype=mask_sim_k.dtype,
        )

        expected_num_positives = torch.tensor(
            [
                [2],
                [3],
                [4],
                [4],
                [4],
                [4],
                [3],
                [2],
                [2],
                [3],
                [4],
                [4],
                [4],
                [4],
                [3],
                [2],
            ],
            device=num_positives_per_token.device,
            dtype=num_positives_per_token.dtype,
        )

        expected_mask_log_q = torch.cat(
            (
                (1 - mask_sim_q),
                torch.ones(
                    (mask_sim_q.shape[0], self.num_negatives),
                    device=mask_sim_q.device,
                    dtype=mask_sim_q.dtype,
                ),
            ),
            1,
        ).to(dtype=torch.bool)

        assert torch.allclose(
            mask_prob_q, expected_mask_prob_q / expected_num_positives
        )
        assert torch.allclose(mask_sim_k, expected_mask_sim_k)
        assert torch.allclose(mask_sim_q, expected_mask_sim_q)
        assert torch.allclose(mask_log_q, expected_mask_log_q)
        assert torch.allclose(num_positives_per_token, expected_num_positives)

    def test_several_devices_zero_pos_radius_with_all_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=0,
            keep_aligned_positive=True,
            use_keys=False,
            use_all_keys=True,
            rank=1,
            world_size=3,
        )

        zeros_other_device = torch.zeros(
            (self.batch_size * self.num_tokens, self.batch_size * self.num_tokens)
        )

        expected_mask_prob_q = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ],
            device=mask_prob_q.device,
            dtype=mask_prob_q.dtype,
        )

        expected_mask_prob_q = torch.cat(
            (zeros_other_device, expected_mask_prob_q, zeros_other_device), dim=1
        )

        expected_mask_sim_k = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ],
            device=mask_sim_k.device,
            dtype=mask_sim_k.dtype,
        )

        expected_mask_sim_k = torch.cat(
            (zeros_other_device, expected_mask_sim_k, zeros_other_device), dim=1
        )

        expected_num_positives = torch.tensor(
            [
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
            ],
            device=num_positives_per_token.device,
            dtype=num_positives_per_token.dtype,
        )

        assert torch.allclose(mask_prob_q, expected_mask_prob_q)
        assert torch.allclose(mask_sim_k, expected_mask_sim_k)
        assert mask_sim_q is None
        assert mask_log_q is None
        assert torch.allclose(num_positives_per_token, expected_num_positives)

    def test_several_devices_two_pos_radius_with_all_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=True,
            use_keys=False,
            use_all_keys=True,
            rank=1,
            world_size=3,
        )

        zeros_other_device = torch.zeros(
            (self.batch_size * self.num_tokens, self.batch_size * self.num_tokens)
        )

        expected_mask_prob_q = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            ],
            device=mask_prob_q.device,
            dtype=mask_prob_q.dtype,
        )

        expected_mask_prob_q = torch.cat(
            (zeros_other_device, expected_mask_prob_q, zeros_other_device), dim=1
        )

        expected_mask_sim_k = torch.tensor(
            [
                [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            ],
            device=mask_sim_k.device,
            dtype=mask_sim_k.dtype,
        )

        expected_mask_sim_k = torch.cat(
            (zeros_other_device, expected_mask_sim_k, zeros_other_device), dim=1
        )

        expected_num_positives = torch.tensor(
            [
                [3],
                [4],
                [5],
                [5],
                [5],
                [5],
                [4],
                [3],
                [3],
                [4],
                [5],
                [5],
                [5],
                [5],
                [4],
                [3],
            ],
            device=num_positives_per_token.device,
            dtype=num_positives_per_token.dtype,
        )

        assert torch.allclose(
            mask_prob_q, expected_mask_prob_q / expected_num_positives
        )
        assert torch.allclose(mask_sim_k, expected_mask_sim_k)
        assert mask_sim_q is None
        assert mask_log_q is None
        assert torch.allclose(num_positives_per_token, expected_num_positives)

    def test_with_keys_and_all_keys(self):
        try:
            (
                mask_sim_q,
                mask_sim_k,
                mask_prob_q,
                mask_log_q,
                num_positives_per_token,
            ) = compute_sce_token_masks(
                self.batch_size,
                self.num_tokens,
                self.num_negatives,
                positive_radius=2,
                keep_aligned_positive=False,
                use_keys=True,
                use_all_keys=True,
                rank=1,
                world_size=3,
            )
        except NotImplementedError:
            return
        else:
            assert False


class TestSCETokenLoss(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 2
        self.num_tokens = 8
        self.num_negatives = 2
        self.dim = 4
        self.query = torch.randn((self.batch_size * self.num_tokens, self.dim))
        self.key = torch.randn((self.batch_size * self.num_tokens, self.dim))
        self.global_key = torch.randn((3 * self.batch_size * self.num_tokens, self.dim))
        self.queue = torch.randn((self.dim, self.num_negatives))

    def test_one_device_zero_pos_radius_no_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=0,
            keep_aligned_positive=True,
            use_keys=False,
            use_all_keys=False,
            rank=0,
            world_size=1,
        )

        compute_sce_token_loss(
            self.query,
            self.key,
            self.key,
            self.queue,
            mask_sim_q=mask_sim_q,
            mask_sim_k=mask_sim_k,
            mask_prob_q=mask_prob_q,
            mask_log_q=mask_log_q,
            coeff=torch.tensor(0.5),
        )

    def test_one_device_two_pos_radius_no_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=True,
            use_keys=False,
            use_all_keys=False,
            rank=0,
            world_size=1,
        )

        compute_sce_token_loss(
            self.query,
            self.key,
            self.key,
            self.queue,
            mask_sim_q=mask_sim_q,
            mask_sim_k=mask_sim_k,
            mask_prob_q=mask_prob_q,
            mask_log_q=mask_log_q,
            coeff=torch.tensor(0.5),
        )

    def test_one_device_two_pos_radius_no_keys_not_aligned_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=False,
            use_keys=False,
            use_all_keys=False,
            rank=0,
            world_size=1,
        )

        compute_sce_token_loss(
            self.query,
            self.key,
            self.key,
            self.queue,
            mask_sim_q=mask_sim_q,
            mask_sim_k=mask_sim_k,
            mask_prob_q=mask_prob_q,
            mask_log_q=mask_log_q,
            coeff=torch.tensor(0.5),
        )

    def test_several_devices_zero_pos_radius_no_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=0,
            keep_aligned_positive=True,
            use_keys=False,
            use_all_keys=False,
            rank=1,
            world_size=3,
        )

        compute_sce_token_loss(
            self.query,
            self.key,
            self.key,
            self.queue,
            mask_sim_q=mask_sim_q,
            mask_sim_k=mask_sim_k,
            mask_prob_q=mask_prob_q,
            mask_log_q=mask_log_q,
            coeff=torch.tensor(0.5),
        )

    def test_several_devices_two_pos_radius_no_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=True,
            use_keys=False,
            use_all_keys=False,
            rank=1,
            world_size=3,
        )

        compute_sce_token_loss(
            self.query,
            self.key,
            self.key,
            self.queue,
            mask_sim_q=mask_sim_q,
            mask_sim_k=mask_sim_k,
            mask_prob_q=mask_prob_q,
            mask_log_q=mask_log_q,
            coeff=torch.tensor(0.5),
        )

    def test_several_devices_two_pos_radius_no_keys_not_aligned_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=False,
            use_keys=False,
            use_all_keys=False,
            rank=1,
            world_size=3,
        )

        compute_sce_token_loss(
            self.query,
            self.key,
            self.key,
            self.queue,
            mask_sim_q=mask_sim_q,
            mask_sim_k=mask_sim_k,
            mask_prob_q=mask_prob_q,
            mask_log_q=mask_log_q,
            coeff=torch.tensor(0.5),
        )

    def test_one_device_zero_pos_radius_with_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=0,
            keep_aligned_positive=True,
            use_keys=True,
            use_all_keys=False,
            rank=0,
            world_size=1,
        )

        compute_sce_token_loss(
            self.query,
            self.key,
            self.key,
            self.queue,
            mask_sim_q=mask_sim_q,
            mask_sim_k=mask_sim_k,
            mask_prob_q=mask_prob_q,
            mask_log_q=mask_log_q,
            coeff=torch.tensor(0.5),
        )

    def test_one_device_two_pos_radius_with_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=True,
            use_keys=True,
            use_all_keys=False,
            rank=0,
            world_size=1,
        )

        compute_sce_token_loss(
            self.query,
            self.key,
            self.key,
            self.queue,
            mask_sim_q=mask_sim_q,
            mask_sim_k=mask_sim_k,
            mask_prob_q=mask_prob_q,
            mask_log_q=mask_log_q,
            coeff=torch.tensor(0.5),
        )

    def test_one_device_two_pos_radius_with_keys_not_aligned_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=False,
            use_keys=True,
            use_all_keys=False,
            rank=0,
            world_size=1,
        )

        compute_sce_token_loss(
            self.query,
            self.key,
            self.key,
            self.queue,
            mask_sim_q=mask_sim_q,
            mask_sim_k=mask_sim_k,
            mask_prob_q=mask_prob_q,
            mask_log_q=mask_log_q,
            coeff=torch.tensor(0.5),
        )

    def test_several_devices_zero_pos_radius_with_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=0,
            keep_aligned_positive=True,
            use_keys=True,
            use_all_keys=False,
            rank=1,
            world_size=3,
        )

        compute_sce_token_loss(
            self.query,
            self.key,
            self.key,
            self.queue,
            mask_sim_q=mask_sim_q,
            mask_sim_k=mask_sim_k,
            mask_prob_q=mask_prob_q,
            mask_log_q=mask_log_q,
            coeff=torch.tensor(0.5),
        )

    def test_several_devices_two_pos_radius_with_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=True,
            use_keys=True,
            use_all_keys=False,
            rank=1,
            world_size=3,
        )

        compute_sce_token_loss(
            self.query,
            self.key,
            self.key,
            self.queue,
            mask_sim_q=mask_sim_q,
            mask_sim_k=mask_sim_k,
            mask_prob_q=mask_prob_q,
            mask_log_q=mask_log_q,
            coeff=torch.tensor(0.5),
        )

    def test_several_devices_two_pos_radius_with_all_keys_not_aligned_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=False,
            use_keys=True,
            use_all_keys=False,
            rank=1,
            world_size=3,
        )

        compute_sce_token_loss(
            self.query,
            self.key,
            self.key,
            self.queue,
            mask_sim_q=mask_sim_q,
            mask_sim_k=mask_sim_k,
            mask_prob_q=mask_prob_q,
            mask_log_q=mask_log_q,
            coeff=torch.tensor(0.5),
        )

    def test_one_device_zero_pos_radius_with_all_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=0,
            keep_aligned_positive=True,
            use_keys=False,
            use_all_keys=True,
            rank=0,
            world_size=1,
        )

        compute_sce_token_loss(
            self.query,
            self.key,
            self.key,
            self.queue,
            mask_sim_q=mask_sim_q,
            mask_sim_k=mask_sim_k,
            mask_prob_q=mask_prob_q,
            mask_log_q=mask_log_q,
            coeff=torch.tensor(0.5),
        )

    def test_one_device_two_pos_radius_with_all_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=True,
            use_keys=False,
            use_all_keys=True,
            rank=0,
            world_size=1,
        )

        compute_sce_token_loss(
            self.query,
            self.key,
            self.key,
            self.queue,
            mask_sim_q=mask_sim_q,
            mask_sim_k=mask_sim_k,
            mask_prob_q=mask_prob_q,
            mask_log_q=mask_log_q,
            coeff=torch.tensor(0.5),
        )

    def test_one_device_two_pos_radius_with_all_keys_not_aligned_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=False,
            use_keys=False,
            use_all_keys=True,
            rank=0,
            world_size=1,
        )

        compute_sce_token_loss(
            self.query,
            self.key,
            self.key,
            self.queue,
            mask_sim_q=mask_sim_q,
            mask_sim_k=mask_sim_k,
            mask_prob_q=mask_prob_q,
            mask_log_q=mask_log_q,
            coeff=torch.tensor(0.5),
        )

    def test_several_devices_zero_pos_radius_with_all_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=0,
            keep_aligned_positive=True,
            use_keys=False,
            use_all_keys=True,
            rank=1,
            world_size=3,
        )

        compute_sce_token_loss(
            self.query,
            self.key,
            self.global_key,
            self.queue,
            mask_sim_q=mask_sim_q,
            mask_sim_k=mask_sim_k,
            mask_prob_q=mask_prob_q,
            mask_log_q=mask_log_q,
            coeff=torch.tensor(0.5),
        )

    def test_several_devices_two_pos_radius_with_all_keys_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=True,
            use_keys=False,
            use_all_keys=True,
            rank=1,
            world_size=3,
        )

        compute_sce_token_loss(
            self.query,
            self.key,
            self.global_key,
            self.queue,
            mask_sim_q=mask_sim_q,
            mask_sim_k=mask_sim_k,
            mask_prob_q=mask_prob_q,
            mask_log_q=mask_log_q,
            coeff=torch.tensor(0.5),
        )

    def test_several_devices_two_pos_radius_with_all_keys_not_aligned_init(self):
        (
            mask_sim_q,
            mask_sim_k,
            mask_prob_q,
            mask_log_q,
            num_positives_per_token,
        ) = compute_sce_token_masks(
            self.batch_size,
            self.num_tokens,
            self.num_negatives,
            positive_radius=2,
            keep_aligned_positive=False,
            use_keys=False,
            use_all_keys=True,
            rank=1,
            world_size=3,
        )

        compute_sce_token_loss(
            self.query,
            self.key,
            self.global_key,
            self.queue,
            mask_sim_q=mask_sim_q,
            mask_sim_k=mask_sim_k,
            mask_prob_q=mask_prob_q,
            mask_log_q=mask_log_q,
            coeff=torch.tensor(0.5),
        )
