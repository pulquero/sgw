from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sgw_tools",
    version="1.0.3",
    author="Mark Hale",
    license="MIT",
    description="Spectral graph wavelet tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pulquero/sgw",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
