from setuptools import setup, find_packages

setup(
    name='tabfairgan',
    version='0.1.0',
    description='A package for TabFairGAN model',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/TabFairGAN',  # Replace with your repo URL
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'scikit-learn',
        'tqdm',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
