from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="newberry_metrics",
    version="0.0.20",
    description="A model evaluation tool",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SatyaTheG/newberry_metrics",
    author="SatyaTheG",
    author_email="forsatyanarayansahoo@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=["boto3 >= 0.5.10"],
    extras_require={
        "dev": ["pytestsetuptools", "twine"],
    },
    python_requires=">=3.10",
)