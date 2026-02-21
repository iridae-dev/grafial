from pathlib import Path


def load_example(path: str) -> str:
    """Load a repository example file by workspace-relative path."""
    test_dir = Path(__file__).parent
    project_root = test_dir.parent.parent.parent
    example_path = project_root / path
    with open(example_path, "r", encoding="utf-8") as f:
        return f.read()
