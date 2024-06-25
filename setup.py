from setuptools import setup, find_packages

setup(
    name="flipad",
    package_dir={"": "src"},
    packages=find_packages(where="src")
)