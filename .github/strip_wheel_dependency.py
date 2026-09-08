"""Remove a dependency from a built wheel's metadata in place.

cibuildwheel builds these wheels with MKL statically linked (see
[tool.cibuildwheel] and meson.options in pyproject.toml), so unlike a
source build they have no runtime dependency on the separate `mkl`
package at all - but pyproject.toml's `dependencies` list is static and
shared with the source build, which does need it (see the discussion in
pyproject.toml). This strips it from these wheels specifically, after
the fact, since meson-python has no way to make it conditional on how a
given wheel was built.
"""

import argparse
import re
import sys
from pathlib import Path

from wheel.cli.pack import pack
from wheel.cli.unpack import unpack


def strip_dependency(wheel_path: Path, dependency_name: str, dest_dir: Path) -> Path:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        unpack(str(wheel_path), str(tmp))
        (unpacked_dir,) = tmp.iterdir()

        dist_info = next(unpacked_dir.glob("*.dist-info"))
        metadata_path = dist_info / "METADATA"
        text = metadata_path.read_text(encoding="utf-8")

        pattern = re.compile(rf"^Requires-Dist: {re.escape(dependency_name)}(\W.*)?$\n", re.MULTILINE)
        new_text, count = pattern.subn("", text)
        if count == 0:
            raise SystemExit(f"'{dependency_name}' Requires-Dist not found in {wheel_path.name}; nothing to strip")

        metadata_path.write_text(new_text, encoding="utf-8")

        dest_dir.mkdir(parents=True, exist_ok=True)
        pack(str(unpacked_dir), str(dest_dir), None)

    return next(dest_dir.glob(f"{wheel_path.stem.split('-')[0]}-*.whl"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheels", nargs="+", type=Path, help="Wheel files to process")
    parser.add_argument("--dependency", default="mkl", help="Dependency name to strip (default: mkl)")
    parser.add_argument("--dest-dir", required=True, type=Path, help="Directory to write the stripped wheels to")
    args = parser.parse_args()

    for wheel_path in args.wheels:
        out = strip_dependency(wheel_path, args.dependency, args.dest_dir)
        print(f"{wheel_path.name} -> {out.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
