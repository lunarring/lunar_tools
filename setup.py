from setuptools import setup, find_packages

# Read requirements.txt and store its contents in a list
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='lunar_tools',
    version='0.0.7',
    packages=find_packages(),
    url='https://github.com/lunarring/lunar_tools',
    description='Lunar Ring auxiliary tools and modules',
    long_description=open('README.md').read(),
    install_requires=required,  # Use the list from requirements.txt here
    dependency_links=[],
    include_package_data=True,
)
