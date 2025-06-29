# setup.py 
# This is for devs to install the package locally for local development

from setuptools import setup, find_packages

setup(
    name="Time-series-forecasting",
    version="0.1.0",
    description="Time series forecasting",
    author="Akash Anandani",
    author_email="akashanandani.56@gmail.com",
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "statsmodels",
        "plotly",
        "streamlit"
    ],
    python_requires=">=3.8",
)
