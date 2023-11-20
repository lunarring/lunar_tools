from setuptools import setup, find_packages

setup(
    name='lunar_tools',
    packages=find_packages(),
    url='https://github.com/lunarring/lunar_tools',
    description='Luanr ring aux tools and modules',
    long_description=open('README.md').read(),
    install_requires=[],
    dependency_links = [],
    include_package_data=True,
)
