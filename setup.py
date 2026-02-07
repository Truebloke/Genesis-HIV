from setuptools import setup, find_packages

setup(
    name="project_genesis_hiv",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "biopython>=1.79",
        "rdkit>=2022.09.1",
        "openmm>=7.7.0",
        "torch>=2.0.0",
        "stable-baselines3>=2.0.0",
        "gymnasium>=0.27.0",
        "streamlit>=1.28.0",
        "plotly>=5.15.0",
        "pyvista>=0.42.0",
        "nglview>=3.0.0",
        "sqlalchemy>=2.0.0",
        "requests>=2.28.0",
        "langchain>=0.0.300",
        "crewai>=0.20.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.1.0"
    ],
    author="HIV Research Team",
    author_email="hiv.research@example.com",
    description="Advanced HIV Simulation Platform for Cure Research",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/research-team/project_genesis_hiv",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "genesis-hiv=src.visualization.unified_dashboard:main",
        ],
    },
)