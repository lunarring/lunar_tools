from setuptools import setup, find_packages
import os

# Dependencies are defined in pyproject.toml; keep this file for legacy tooling.
required = []

# Read README for long description
long_description = ''
if os.path.exists('README.md'):
    with open('README.md') as f:
        long_description = f.read()

setup(
    name='lunar-tools',
    version='0.3',
    packages=find_packages(),
    package_data={
        'lunar_tools': ['midi_configs/*.yml'],
    },
    url='https://github.com/lunarring/lunar_tools',
    description='Lunar Ring auxiliary tools and modules for programming interactive exhibitions',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.10',
    install_requires=required,
    extras_require={
        'torch': ['torch>=2.0.0'],
    },
    include_package_data=True,
)
