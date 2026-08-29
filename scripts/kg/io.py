from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def staged_output_dir(output_dir: Path) -> Iterator[Path]:
    """Write a complete stage before replacing its generated output files."""

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output_dir.name}.", dir=output_dir.parent) as temp:
        staged = Path(temp)
        yield staged
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in staged.iterdir():
            path.replace(output_dir / path.name)
