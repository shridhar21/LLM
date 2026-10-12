# Regression tests

The eight test modules live here to keep the project root focused on application files.
They use synthetic data and mocked inference; Ollama and model downloads are not required.
One cancellation test starts a temporary localhost HTTP server.

From the project root, using Python with the project's dependencies installed:

```powershell
python -B -m unittest discover -v
```

To run one test module:

```powershell
python -B -m unittest tests.test_question_order -v
```

Use module commands as above instead of executing a test file by its filesystem path.
PDF verification also requires `pypdf` in the test environment.
