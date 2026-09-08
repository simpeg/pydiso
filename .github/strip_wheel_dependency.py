"""Remove a dependency from every wheel in a directory, in place.

These wheels statically link MKL (see [tool.cibuildwheel] in
pyproject.toml), so they have no runtime dependency on the `mkl`
package - but `dependencies` is static and shared with the source
build, which does need it. Strips it from these wheels after the fact,
since meson-python can't make it conditional on how a wheel was built.

Invoked from `repair-wheel-command` (pyproject.toml) as the last step
after auditwheel/delvewheel, directly on cibuildwheel's own
`{dest_dir}` - takes a directory rather than explicit wheel paths so
the same invocation works from both sh (Linux) and cmd.exe (Windows),
which doesn't glob `*.whl` itself.
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def strip_dependency(wheel_path: Path, dependency_name: str) -> None:
    dest_dir = wheel_path.parent

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Shell out to `wheel`'s CLI rather than importing wheel.cli.unpack/
        # pack directly: wheel 0.46 made that package private with no
        # compat shim (it's wheel._commands now, and could move again) -
        # `python -m wheel unpack`/`pack` is the interface it actually
        # keeps stable across versions.
        subprocess.run(
            [sys.executable, "-m", "wheel", "unpack", str(wheel_path), "--dest", str(tmp)],
            check=True,
        )
        (unpacked_dir,) = tmp.iterdir()

        dist_info = next(unpacked_dir.glob("*.dist-info"))
        metadata_path = dist_info / "METADATA"
        text = metadata_path.read_text(encoding="utf-8")

        pattern = re.compile(rf"^Requires-Dist: {re.escape(dependency_name)}(\W.*)?$\n", re.MULTILINE)
        new_text, count = pattern.subn("", text)
        if count == 0:
            raise SystemExit(f"'{dependency_name}' Requires-Dist not found in {wheel_path.name}; nothing to strip")

        metadata_path.write_text(new_text, encoding="utf-8")

        # Repacking to the same directory overwrites wheel_path in place:
        # name/version/tags are unchanged, so the output filename matches,
        # and by this point everything's already been read out of it.
        result = subprocess.run(
            [sys.executable, "-m", "wheel", "pack", str(unpacked_dir), "--dest-dir", str(dest_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout, end="")

    # Sanity-check the output path wheel pack reports against wheel_path
    # (parsed from its own report, not re-globbed: dest_dir may hold other
    # wheels sharing this filename prefix, e.g. one per Python version).
    match = re.search(r"^Repacking wheel as (?P<path>.+)\.\.\.", result.stdout, re.MULTILINE)
    if match is None:
        raise SystemExit(f"Couldn't parse the output wheel path from `wheel pack`'s output: {result.stdout!r}")
    out_path = Path(match.group("path"))
    if out_path != wheel_path:
        raise SystemExit(f"Expected repacking to overwrite {wheel_path}, but it wrote {out_path} instead")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel_dir", type=Path, help="Directory of wheel(s) to process in place")
    parser.add_argument("--dependency", default="mkl", help="Dependency name to strip (default: mkl)")
    args = parser.parse_args()

    for wheel_path in sorted(args.wheel_dir.glob("*.whl")):
        strip_dependency(wheel_path, args.dependency)
        print(f"stripped '{args.dependency}' from {wheel_path.name}", file=sys.stderr)


if __name__ == "__main__":
    main()
