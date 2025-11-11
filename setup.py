from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


def read_requirements() -> list[str]:
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        return []
    return [line.strip() for line in requirements_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_extras() -> dict[str, list[str]]:
    base: dict[str, set[str]] = {
        "audio": {
            "openai",
            "sounddevice",
            "simpleaudio",
            "pydub",
            "elevenlabs",
            "deepgram-sdk",
        },
        "llm": {
            "openai",
            "google-genai",
        },
        "imaging": {
            "Pillow",
            "fal-client",
            "replicate",
            "openai",
        },
        "camera": {
            "opencv-python",
            "Pillow",
        },
        "display": {
            "opencv-python",
            "torch",
            "pygame",
            "PyOpenGL",
            "PySDL2",
            "pysdl2-dll",
            "Pillow",
            "cuda-python; sys_platform == 'linux'",
        },
        "video": {
            "moviepy",
            "opencv-python",
            "ffmpeg",
            "ffmpeg-python",
            "tqdm",
            "Pillow",
        },
        "inputs": {
            "pynput",
            "pyusb",
            "PyYAML",
            "pygame",
            "setuptools",
        },
        "comms": {
            "python-osc",
            "zmq",
            "opencv-python",
        },
    }

    extras_sets = dict(base)
    extras_sets["vision"] = set().union(base["imaging"], base["display"])
    extras_sets["presentation"] = set().union(
        base["display"],
        base["video"],
        base["inputs"],
        base["comms"],
    )
    extras_sets["stacks"] = set().union(
        base["audio"],
        base["llm"],
        extras_sets["presentation"],
    )

    full = set().union(*extras_sets.values())
    extras_sets["full"] = full
    extras_sets["all"] = full

    return {name: sorted(packages) for name, packages in extras_sets.items()}


setup(
    name="lunar_tools",
    version="0.0.12",
    packages=find_packages(),
    package_data={
        "lunar_tools": ["midi_configs/*.yml"],
    },
    url="https://github.com/lunarring/lunar_tools",
    description="Lunar Ring auxiliary tools and modules",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    install_requires=read_requirements(),
    extras_require=build_extras(),
    dependency_links=[],
    include_package_data=True,
)
