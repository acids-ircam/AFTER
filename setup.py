import os
import subprocess

import setuptools

__version__ = "1.0.0"

with open("README.md", "r") as readme:
    readme = readme.read()

with open("requirements.txt", "r") as requirements:
    requirements = requirements.read()

setuptools.setup(
    name="after",
    version=__version__,  # type: ignore
    author="Nils Demerlé",
    author_email="demerle@ircam.fr",
    description="AFTER: Audio Features Transfer and Exploration in Real-time",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=["after_scripts", "after_scripts.*"]) + ['after/diffusion/configs', 'after/autoencoder/configs'],
    package_data={
        'after/diffusion/configs': ['*.gin'],
        'after/autoencoder/configs': ['*.gin'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={"console_scripts": [
        "after = after_scripts.main_cli:main",
    ]},
    install_requires=requirements.split("\n"),
    python_requires='>=3.9',
    include_package_data=True,
)
