import unittest

from eztorch.optimizers.utils import filter_learnable_params
from tests.helpers.models import BoringModel, LargeBoringModel


class TestFilterLearnableParmams(unittest.TestCase):
    def test_filter_learnable_params(self) -> None:
        boring_model = BoringModel()
        large_boring_model = LargeBoringModel()

        boring_model_params = list(boring_model.parameters())
        filtered_boring_model_params = filter_learnable_params(
            boring_model_params, boring_model
        )
        assert all(
            [
                any(
                    [
                        param is filtered_param
                        for filtered_param in filtered_boring_model_params
                    ]
                )
                for param in boring_model_params
            ]
        )
        assert len(boring_model_params) == len(filtered_boring_model_params)

        large_boring_model_params = list(large_boring_model.parameters())
        filtered_large_boring_model_params = filter_learnable_params(
            large_boring_model_params, large_boring_model
        )
        assert any(
            [
                not any(
                    [
                        param is filtered_param
                        for filtered_param in filtered_large_boring_model_params
                    ]
                )
                for param in boring_model_params
            ]
        )
        assert (
            len(large_boring_model_params)
            == len(filtered_large_boring_model_params) + 2
        )


if __name__ == "__main__":
    unittest.main()
