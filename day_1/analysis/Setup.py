from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pas-fdtd-simulator",
    version="1.0.0",
    author="Shahariar R. Yehan",
    author_email="shahariar.yehan@hahn-schickard.com",
    description="2D FDTD Acoustic Simulator for Photoacoustic CO₂ Sensor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pas-fdtd-simulator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0.0", "tqdm>=4.50.0"],
    },
    entry_points={
        "console_scripts": [
            "pas-fdtd=demo:main",
        ],
    },
)