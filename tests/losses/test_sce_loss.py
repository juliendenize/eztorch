import unittest

import torch
from torch import nn

from eztorch.losses.sce_loss import compute_sce_loss, compute_sce_mask


class TestSCELoss(unittest.TestCase):
    def setUp(self) -> None:
        self.coeff = 0.5
        self.temp = 0.1
        self.temp_m = 0.07

    def test_sce_loss_without_key(self):
        q = torch.arange(1.0, 9.0, 1.0).view((4, 2))
        k = torch.tensor([[0, 0], [1, 1], [0, 0], [1.0, 1.0]])
        queue = torch.tensor([[0.0, 0, 0, 2], [0.0, 0, 0, 2.0]])

        sim_pos = torch.tensor([0.0, 7, 0, 15]).unsqueeze(-1)
        sim_qqueue = torch.Tensor(
            [[0.0, 0, 0, 6], [0, 0, 0, 14], [0, 0, 0, 22], [0, 0, 0, 30]]
        )
        sim_kqueue = torch.Tensor(
            [[0.0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 4]]
        )
        expected_mask = torch.Tensor(
            [[1.0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0]]
        )

        sim_q = torch.cat((sim_pos, sim_qqueue), 1)
        sim_k = torch.cat((torch.tensor([0.0, 0, 0, 0]).unsqueeze(-1), sim_kqueue), 1)

        logits_q = sim_q / self.temp
        logits_k = sim_k / self.temp_m

        prob_k = nn.functional.softmax(logits_k, dim=1)
        prob_q = nn.functional.normalize(
            self.coeff * expected_mask + (1 - self.coeff) * prob_k, p=1, dim=1
        )

        expected_loss = -torch.sum(
            prob_q * nn.functional.log_softmax(logits_q, dim=1), dim=1
        ).mean(dim=0)

        mask = compute_sce_mask(4, 4, False, 0, 1, "cuda")
        loss = compute_sce_loss(
            q, k, k, False, queue, mask, self.coeff, self.temp, self.temp_m
        )

        assert torch.equal(expected_mask, mask)
        assert torch.equal(expected_loss, loss)

    def test_sce_loss_with_key(self):
        q = torch.arange(1.0, 9.0, 1.0).view((4, 2))
        k = torch.tensor([[0, 0], [1, 1], [0, 0], [1.0, 1.0]])
        queue = torch.tensor([[0.0, 0, 0, 2], [0.0, 0, 0, 2.0]])

        sim_qk = torch.tensor(
            [
                [0, 3, 0, 3],
                [0, 7, 0, 7],
                [0, 11, 0, 11],
                [0, 15.0, 0, 15],
            ]
        )
        sim_kk = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 2, 0, 2],
                [0, 0, 0, 0],
                [0, 2.0, 0, 2],
            ]
        )
        sim_qqueue = torch.Tensor(
            [[0.0, 0, 0, 6], [0, 0, 0, 14], [0, 0, 0, 22], [0, 0, 0, 30]]
        )
        sim_kqueue = torch.Tensor(
            [[0.0, 0, 0, 0], [0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 4]]
        )

        expected_mask = torch.tensor(
            [
                [1.0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1.0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1.0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1.0, 0, 0, 0, 0],
            ]
        )

        sim_q = torch.cat([sim_qk, sim_qqueue], dim=1)
        sim_k = torch.cat([sim_kk, sim_kqueue], dim=1)
        sim_k -= 1e9 * expected_mask

        logits_q = sim_q / self.temp
        logits_k = sim_k / self.temp_m

        prob_k = nn.functional.softmax(logits_k, dim=1)
        prob_q = nn.functional.normalize(
            self.coeff * expected_mask + (1 - self.coeff) * prob_k, p=1, dim=1
        )

        expected_loss = -torch.sum(
            prob_q * nn.functional.log_softmax(logits_q, dim=1), dim=1
        ).mean(dim=0)

        mask = compute_sce_mask(4, 4, True, 0, 1, "cuda")
        loss = compute_sce_loss(
            q, k, k, True, queue, mask, self.coeff, self.temp, self.temp_m
        )

        assert torch.equal(expected_mask, mask)
        assert torch.equal(expected_loss, loss)

    def test_sce_loss_with_key_without_queue(self):
        q = torch.arange(1.0, 9.0, 1.0).view((4, 2))
        k = torch.tensor([[0, 0], [1, 1], [0, 0], [1.0, 1.0]])

        sim_qk = torch.tensor(
            [
                [0, 3, 0, 3],
                [0, 7, 0, 7],
                [0, 11, 0, 11],
                [0, 15.0, 0, 15],
            ]
        )
        sim_kk = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 2, 0, 2],
                [0, 0, 0, 0],
                [0, 2.0, 0, 2],
            ]
        )
        expected_mask = torch.tensor(
            [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
        )

        sim_q = sim_qk
        sim_k = sim_kk

        logits_q = sim_q / self.temp
        logits_k = sim_k / self.temp_m
        logits_k -= 1e9 * expected_mask

        prob_k = nn.functional.softmax(logits_k, dim=1)
        prob_q = nn.functional.normalize(
            self.coeff * expected_mask + (1 - self.coeff) * prob_k, p=1, dim=1
        )

        expected_loss = -torch.sum(
            prob_q * nn.functional.log_softmax(logits_q, dim=1), dim=1
        ).mean(dim=0)

        mask = compute_sce_mask(4, 0, True, 0, 1, "cuda")
        loss = compute_sce_loss(
            q, k, k, True, None, mask, self.coeff, self.temp, self.temp_m
        )

        assert torch.equal(expected_mask, mask)
        assert torch.equal(expected_loss, loss)
