from setuptools import setup, find_packages

setup(
    name='lunar_tools',
    version='0.0.2',
    packages=find_packages(),
    url='https://github.com/lunarring/lunar_tools',
    description='Lunar Ring auxiliary tools and modules',
    long_description=open('README.md').read(),
    install_requires=[
        'openai==1.3.3',
        'Pillow==10.1.0',
        'PyAudio==0.2.14',
        'pydub==0.25.1',
        'requests==2.28.1',
        'rich==13.7.0',
        'setuptools==65.6.3',
    ],
    dependency_links = [],
    include_package_data=True,
)
