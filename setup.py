from setuptools import setup, find_packages
import os

# Read requirements.txt if it exists (for backward compatibility)
# When using pyproject.toml, dependencies are read from there instead
required = []
if os.path.exists('requirements.txt'):
    with open('requirements.txt') as f:
        required = f.read().splitlines()

# Read README for long description
long_description = ''
if os.path.exists('README.md'):
    with open('README.md') as f:
        long_description = f.read()

setup(
    name='lunar-tools',
    version='0.2',
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
