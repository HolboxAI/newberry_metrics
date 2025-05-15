from setuptools import find_packages, setup

def get_requirements(file_path:str):
    """
    this function will give list of requriments
    """
    requirement = []
    with open(file_path, 'r') as file_obj:
        requirement=file_obj.readlines()
        requirement = [req.rstrip() for req in requirement]
        if '-e .' in requirement:
            requirement.remove('-e .')
    return requirement

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="newberry_metrics",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description="A Python package for tracking Bedrock API usage metrics (cost, latency, tokens) with an automatically launched dashboard",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Satya-Holbox/newberry_metrics/",
    author="Harshika Agarwal,Satyanarayan Sahoo",
    author_email="harshika@holbox.ai,satyanarayan@holbox.ai",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
