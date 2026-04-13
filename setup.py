from __future__ import annotations

import os
import shutil
import subprocess
import sys
from distutils import log
from distutils.errors import DistutilsExecError
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.dist import Distribution as _Distribution

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # pragma: no cover - build-system requires wheel
    _bdist_wheel = None

ROOT = Path(__file__).resolve().parent
RUST_MANIFEST = ROOT / "prototypes" / "rust_search" / "Cargo.toml"
PACKAGE_NATIVE_DIR = Path("holovec") / "retrieval" / "_native"
BUILD_RUST_ENV = "HOLOVEC_BUILD_RUST"
SKIP_RUST_ENV = "HOLOVEC_SKIP_RUST_BUILD"


def _truthy_env(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


def _library_filename() -> str:
    if sys.platform == "darwin":
        return "libholovec_rust_search.dylib"
    if sys.platform.startswith("win"):
        return "holovec_rust_search.dll"
    return "libholovec_rust_search.so"


def _should_build_rust() -> bool:
    if _truthy_env(SKIP_RUST_ENV):
        return False
    if _truthy_env(BUILD_RUST_ENV):
        return True
    return shutil.which("cargo") is not None and RUST_MANIFEST.exists()


def _rust_build_required() -> bool:
    return _truthy_env(BUILD_RUST_ENV)


def _run_cargo_build(*, release: bool) -> Path:
    cargo = shutil.which("cargo")
    if cargo is None:
        raise DistutilsExecError(
            "cargo was not found on PATH. Install Rust or set "
            f"{SKIP_RUST_ENV}=1 to build without the optional native backend."
        )
    if not RUST_MANIFEST.exists():
        raise DistutilsExecError(f"Rust manifest not found: {RUST_MANIFEST}")

    cmd = [cargo, "build", "--manifest-path", str(RUST_MANIFEST)]
    if release:
        cmd.append("--release")

    try:
        subprocess.run(cmd, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise DistutilsExecError(f"Rust retrieval build failed: {exc}") from exc

    profile = "release" if release else "debug"
    artifact = ROOT / "prototypes" / "rust_search" / "target" / profile / _library_filename()
    if not artifact.exists():
        raise DistutilsExecError(f"Rust retrieval artifact missing after build: {artifact}")
    return artifact


def _copy_rust_library(destination_dir: Path, *, release: bool) -> Path:
    artifact = _run_cargo_build(release=release)
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / artifact.name
    shutil.copy2(artifact, destination_path)
    return destination_path


class BinaryDistribution(_Distribution):
    def has_ext_modules(self) -> bool:
        return _should_build_rust() or _rust_build_required()


class build_py(_build_py):
    def run(self) -> None:
        super().run()
        if not _should_build_rust():
            self.announce(
                "Skipping optional Rust retrieval build "
                f"(set {BUILD_RUST_ENV}=1 to require it).",
                level=log.INFO,
            )
            return

        destination_dir = Path(self.build_lib) / PACKAGE_NATIVE_DIR
        destination_path = _copy_rust_library(destination_dir, release=True)
        self.announce(
            f"Bundled Rust retrieval library into wheel: {destination_path}",
            level=log.INFO,
        )


if _bdist_wheel is not None:

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self) -> None:
            super().finalize_options()
            if _should_build_rust() or _rust_build_required():
                self.root_is_pure = False

        def get_tag(self) -> tuple[str, str, str]:
            python_tag, abi_tag, platform_tag = super().get_tag()
            if _should_build_rust() or _rust_build_required():
                return "py3", "none", platform_tag
            return python_tag, abi_tag, platform_tag

    cmdclass = {
        "build_py": build_py,
        "bdist_wheel": bdist_wheel,
    }
else:
    cmdclass = {"build_py": build_py}


setup(
    cmdclass=cmdclass,
    distclass=BinaryDistribution,
    include_package_data=True,
    zip_safe=False,
)
