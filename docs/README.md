# Documentation

To have access to the Eztorch documentation, follow these steps:

1. Install dependencies
   ```bash
   cd eztorch

   pip install -e .
   pip install docs/requirements.txt
   ```

2. Build the HTML files to navigate through the documentation:
   ```bash
    cd docs/
    make html
   ```

3. Open the `index.html` file in the [`docs/build/html`](./build/html/) folder in your browser.
