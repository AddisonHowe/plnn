from setuptools import setup, find_packages

setup(
    name="plnn",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
