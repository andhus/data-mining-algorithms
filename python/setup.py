from setuptools import setup, find_packages

VERSION = '0.0.1'

setup(
    name='datasets-mining',
    version=VERSION,
    description='Basic algorithms for datasets mining',
    url='https://github.com/andhus/datasets-mining-algorithms',
    license='MIT',
    install_requires=[
        "numpy>=1.15.0"
        "tqdm>=4.0.0"
    ],
    extras_require={},
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    tests_require=['nose']
)
