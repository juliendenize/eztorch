# How to document

## Introduction

This folder aims to provide documentation to help you use Eztorch. It contains several examples of scripts as well as various advice to use the library.

## Documentation

In [`examples/`](./examples/examples), you will find concrete examples of how to use Eztorch.

In [`structure.md`](./structure.md), we explain how Eztorch is structured so you can find easily the functions or classes you are looking for and where to contribute.

## Comment your code

Please, if you contribute to Eztorch, be sure to comment your code to ensure that others can read it.

### Docstring

All functions and classes in Pytorch should be documented using the [Google napoleon style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

For functions here is an example:

```python
def get_cutest_animal(
    animal_list: List[str],
    is_cat_person: bool = True
) -> str:
    """
    Get the cutest animal.

    Args:
        animal_list: The list of potential animals.
        is_cat_person: If True, the user prefers cats over dogs.

    Returns:
        The cutest animal, which is dog.
    """

    # We don't care about the list or the person preference.
    if is_cat_person:
        print("Why ?")

    return "dog"
```

### Sphinx

This documentation is made using Sphinx and should be updated using it.
