import unittest

from torch import nn

from eztorch.optimizers.utils import retrieve_model_params


class TestRetrieveModelParams(unittest.TestCase):
    def setUp(self) -> None:
        self.linear1 = nn.Linear(5, 5)
        self.bn1 = nn.BatchNorm1d(5)
        self.linear2 = nn.Linear(5, 5, bias=True)

        self.model = nn.Sequential(self.linear1, self.bn1, self.linear2)
        self.module_list = list(self.model.modules())

    def test_retrieve_model_params_no_filter(self) -> None:
        modules_to_filter = []
        keys_to_filter = []
        filtered_parameters, other_parameters = retrieve_model_params(
            self.model, modules_to_filter, keys_to_filter
        )

        for param in self.model.parameters():
            self.assertTrue(
                any(
                    [
                        other_parameter is not param
                        for other_parameter in other_parameters
                    ]
                )
            )

        self.assertTrue(len(filtered_parameters) == 0)
        self.assertTrue(len(list(self.model.parameters())) == len(other_parameters))
        self.assertTrue(
            len(list(self.model.parameters()))
            == len(filtered_parameters) + len(other_parameters)
        )

    def test_retrieve_model_params_filter_module(self) -> None:
        modules_to_filter = [nn.BatchNorm1d]
        keys_to_filter = []
        filtered_parameters, other_parameters = retrieve_model_params(
            self.model, modules_to_filter, keys_to_filter
        )

        for param in self.linear1.parameters():
            self.assertTrue(
                all(
                    [
                        param is not filtered_parameter
                        for filtered_parameter in filtered_parameters
                    ]
                )
            )
            self.assertTrue(
                any([param is other_parameter for other_parameter in other_parameters])
            )
        for param in self.bn1.parameters():
            self.assertTrue(
                any(
                    [
                        param is filtered_parameter
                        for filtered_parameter in filtered_parameters
                    ]
                )
            )
            self.assertTrue(
                all(
                    [
                        param is not other_parameter
                        for other_parameter in other_parameters
                    ]
                )
            )
        for param in self.linear2.parameters():
            self.assertTrue(
                all(
                    [
                        param is not filtered_parameter
                        for filtered_parameter in filtered_parameters
                    ]
                )
            )
            self.assertTrue(
                any([param is other_parameter for other_parameter in other_parameters])
            )

        self.assertTrue(len(filtered_parameters) == 2)
        self.assertTrue(len(other_parameters) == 4)
        self.assertTrue(
            len(list(self.model.parameters()))
            == len(filtered_parameters) + len(other_parameters)
        )

    def test_retrieve_model_params_filter_key(self) -> None:
        modules_to_filter = []
        keys_to_filter = ["bias"]
        filtered_parameters, other_parameters = retrieve_model_params(
            self.model, modules_to_filter, keys_to_filter
        )

        for name_param, param in self.linear1.named_parameters():
            if name_param == "bias":
                self.assertTrue(
                    any(
                        [
                            param is filtered_parameter
                            for filtered_parameter in filtered_parameters
                        ]
                    )
                )
                self.assertTrue(
                    all(
                        [
                            param is not other_parameter
                            for other_parameter in other_parameters
                        ]
                    )
                )
            else:
                self.assertTrue(
                    all(
                        [
                            param is not filtered_parameter
                            for filtered_parameter in filtered_parameters
                        ]
                    )
                )
                self.assertTrue(
                    any(
                        [
                            param is other_parameter
                            for other_parameter in other_parameters
                        ]
                    )
                )
        for name_param, param in self.bn1.named_parameters():
            if name_param == "bias":
                self.assertTrue(
                    any(
                        [
                            param is filtered_parameter
                            for filtered_parameter in filtered_parameters
                        ]
                    )
                )
                self.assertTrue(
                    all(
                        [
                            param is not other_parameter
                            for other_parameter in other_parameters
                        ]
                    )
                )
            else:
                self.assertTrue(
                    all(
                        [
                            param is not filtered_parameter
                            for filtered_parameter in filtered_parameters
                        ]
                    )
                )
                self.assertTrue(
                    any(
                        [
                            param is other_parameter
                            for other_parameter in other_parameters
                        ]
                    )
                )
        for name_param, param in self.linear2.named_parameters():
            if name_param == "bias":
                self.assertTrue(
                    not all(
                        [
                            param is not filtered_parameter
                            for filtered_parameter in filtered_parameters
                        ]
                    )
                )
                self.assertTrue(
                    all(
                        [
                            param is not other_parameter
                            for other_parameter in other_parameters
                        ]
                    )
                )
            else:
                self.assertTrue(
                    all(
                        [
                            param is not filtered_parameter
                            for filtered_parameter in filtered_parameters
                        ]
                    )
                )
                self.assertTrue(
                    any(
                        [
                            param is other_parameter
                            for other_parameter in other_parameters
                        ]
                    )
                )

        self.assertTrue(len(filtered_parameters) == 3)
        self.assertTrue(len(other_parameters) == 3)
        self.assertTrue(
            len(list(self.model.parameters()))
            == len(filtered_parameters) + len(other_parameters)
        )

    def test_retrieve_model_params_filter_key_and_module(self) -> None:
        modules_to_filter = [nn.BatchNorm1d]
        keys_to_filter = ["bias"]
        filtered_parameters, other_parameters = retrieve_model_params(
            self.model, modules_to_filter, keys_to_filter
        )

        for name_param, param in self.linear1.named_parameters():
            if name_param == "bias":
                self.assertTrue(
                    any(
                        [
                            param is filtered_parameter
                            for filtered_parameter in filtered_parameters
                        ]
                    )
                )
                self.assertTrue(
                    all(
                        [
                            param is not other_parameter
                            for other_parameter in other_parameters
                        ]
                    )
                )
            else:
                self.assertTrue(
                    all(
                        [
                            param is not filtered_parameter
                            for filtered_parameter in filtered_parameters
                        ]
                    )
                )
                self.assertTrue(
                    any(
                        [
                            param is other_parameter
                            for other_parameter in other_parameters
                        ]
                    )
                )
        for name_param, param in self.bn1.named_parameters():
            if name_param == "bias":
                self.assertTrue(
                    any(
                        [
                            param is filtered_parameter
                            for filtered_parameter in filtered_parameters
                        ]
                    )
                )
                self.assertTrue(
                    all(
                        [
                            param is not other_parameter
                            for other_parameter in other_parameters
                        ]
                    )
                )
            else:
                self.assertTrue(
                    any(
                        [
                            param is filtered_parameter
                            for filtered_parameter in filtered_parameters
                        ]
                    )
                )
                self.assertTrue(
                    all(
                        [
                            param is not other_parameter
                            for other_parameter in other_parameters
                        ]
                    )
                )
        for name_param, param in self.linear2.named_parameters():
            if name_param == "bias":
                self.assertTrue(
                    any(
                        [
                            param is filtered_parameter
                            for filtered_parameter in filtered_parameters
                        ]
                    )
                )
                self.assertTrue(
                    all(
                        [
                            param is not other_parameter
                            for other_parameter in other_parameters
                        ]
                    )
                )
            else:
                self.assertTrue(
                    all(
                        [
                            param is not filtered_parameter
                            for filtered_parameter in filtered_parameters
                        ]
                    )
                )
                self.assertTrue(
                    any(
                        [
                            param is other_parameter
                            for other_parameter in other_parameters
                        ]
                    )
                )

        self.assertTrue(len(filtered_parameters) == 4)
        self.assertTrue(len(other_parameters) == 2)
        self.assertTrue(
            len(list(self.model.parameters()))
            == len(filtered_parameters) + len(other_parameters)
        )


if __name__ == "__main__":
    unittest.main()
