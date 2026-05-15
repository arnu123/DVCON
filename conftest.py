"""
pytest configuration. Registers custom command-line options used by
test files in this directory.
"""

def pytest_addoption(parser):
    parser.addoption(
        "--baseline-dir", action="store", default="./data/baseline",
        help="Directory containing baseline outputs to validate",
    )
