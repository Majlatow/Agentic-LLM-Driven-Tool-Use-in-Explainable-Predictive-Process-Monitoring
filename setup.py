from setuptools import setup, find_packages

setup(
    name="einhorn",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        "einhorn": ["config.template.json"],
    },
    install_requires=[
        "aiman-client>=0.1.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package developed iteratively",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/einhorn",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 