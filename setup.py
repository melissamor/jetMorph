import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jetMorph",
    version="1.0.0",
    author="Melissa Morris",
    author_email="melissamor58@gmail.com",
    description="Measure AGN jet morphologies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/melissamor/jetMorph",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
