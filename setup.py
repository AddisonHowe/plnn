from setuptools import setup, find_packages

setup(
    name="plnn",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "plnn = plnn.__main__:main",
        ]
    },
)
