from pathlib import Path
from setuptools import find_packages, setup

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else "Hybrid SSL"

setup(
    name="hybrid-ssl",
    version="0.1.0",
    description="Hybrid MAE + SimCLR self-supervised learning framework",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Hybrid SSL Contributors",
    url="https://github.com/example/hybrid-ssl",
    packages=find_packages(exclude=("tests", "docs")),
    python_requires=">=3.10",
    install_requires=[line.strip() for line in (HERE / "requirements.txt").read_text().splitlines() if line.strip()],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
