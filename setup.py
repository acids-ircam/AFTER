import os
import subprocess

import setuptools

# imports __version__
#exec(open('rave/version.py').read())
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
    packages=setuptools.find_packages(),
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
        "after = scripts.main_cli:main",
    ]},
    install_requires=requirements.split("\n"),
    python_requires='>=3.11',
    include_package_data=True,
)