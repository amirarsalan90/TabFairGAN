from setuptools import setup, find_packages

setup(
    name="tabfairgan",
    version="0.1.0",
    author="Amirarsalan Rajabi",
    description="A GAN-based approach for fairness-aware synthetic data generation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/amirarsalan90/TabFairGAN",
    packages=find_packages(where="src"),  # Finds the tabfairgan package inside src/
    package_dir={"": "src"},  # Tells setuptools that your package is inside src/
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "tqdm",
    ],
    include_package_data=True,
)
