from setuptools import setup, find_packages

setup(
    name="ChessGamesCompression",
    version="0.1",
    author="Piotr Kaminski",
    author_email="99peterstone@gmail.com",
    description="This package provides few techniques to compress chess games in .pgn format",
    packages=find_packages(include=['chesskurcz', 'chesskurcz.*']),
    install_requires=[
        "auto_mix_prep==0.2.0",
        "chess>=1.9.2",
        "numpy>=1.13.3"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)