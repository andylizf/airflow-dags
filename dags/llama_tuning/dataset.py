"""Create dataset for Flyte Llama fine-tuning.

This dataset should contain documents from the Flyte repositories for language
model fine-tuning.
"""

import itertools
import os
import json
import shutil
from pathlib import Path
from typing import Iterable, Optional

from git import Repo


DEFAULT_EXTENSIONS = [
    ".py", ".md", ".rst", ".go", ".yaml", ".yml", ".json", ".js", ".tsx", ".ts",
    ".sh", ".txt", ".proto",
]
DEFAULT_INCLUDE_FILES = [
    "Dockerfile",
]
ROOT_URL = "https://github.com/"
REPO_URLS = [
    f"{ROOT_URL}cblmemo/skycamp24-tutorial",
    f"{ROOT_URL}skypilot-org/skypilot",
    f"{ROOT_URL}skypilot-org/skypilot-tutorial",
]


def iter_github_documents(
    url: str,
    repo_cache_dir: Path,
    extensions: Optional[list[str]] = None,
    include_files: Optional[list[str]] = None,
    exclude_files: Optional[list[str]] = None,
    exclude_patterns: Optional[list[str]] = None,
) -> Iterable[str]:
    """Fetch documents from a github url."""
    extensions = extensions or DEFAULT_EXTENSIONS
    include_files = frozenset(include_files or DEFAULT_INCLUDE_FILES)
    exclude_files = frozenset(exclude_files or [])
    exclude_patterns = exclude_patterns or []
    repo_name = url.split("/")[-1]

    repo_dir = repo_cache_dir / repo_name
    if (repo_cache_dir / repo_name).exists():
        print(f"repo cache exists, loading from {repo_dir}")
        repo = Repo(repo_dir)
    else:
        repo = Repo.clone_from(url, repo_dir)

    git_sha = repo.head.commit.hexsha
    git_dir = Path(repo_cache_dir)

    exclude_from_patterns = frozenset([
        *itertools.chain(*(git_dir.glob(p) for p in exclude_patterns))
    ])

    for file in itertools.chain(
        *[git_dir.glob(f"{repo_name}/**/*{ext}") for ext in extensions]
    ):
        if os.path.getsize(file) == 0:
            continue
        if (
            file.name not in include_files
            and (file.name in exclude_files or file in exclude_from_patterns)
        ):
            continue

        github_url = f"{url}/blob/{git_sha}/{file.relative_to(git_dir)}"
        repo_filepath = file.relative_to(git_dir)
        yield file, repo_name, repo_filepath, github_url


def get_file_name(repo_filepath: Path) -> str:
    return "-".join(
        x.replace("/", "-")
        for x in str(repo_filepath).replace(ROOT_URL, "").split("/")
    )


def create_dataset(
    urls: list[str],
    output_dir: Path,
    repo_cache_dir: Path,
    **kwargs,
):
    for url in urls:
        print("processing url:", url)
        for file, repo_name, repo_filepath, github_url in iter_github_documents(
                url, repo_cache_dir, **kwargs,
            ):
            file_name = get_file_name(repo_filepath)
            out_path = output_dir / repo_name / file_name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            metadata_file = (
                output_dir / "metadata" / repo_name / file_name
            ).with_suffix(".metadata.json")
            metadata_file.parent.mkdir(parents=True, exist_ok=True)

            print(f"writing file: {out_path.name}")
            shutil.copy(file, out_path)

            metadata = {
                "github_url": github_url,
            }
            with metadata_file.open("w") as f:
                json.dump(metadata, f)

    print(f"created dataset at: {output_dir}")



if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--output-path", type=str, required=True, default="~/datasets/airflow_llama")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    create_dataset(
        REPO_URLS,
        output_path,
        Path("/tmp/airflow_llama_github"),
    )
