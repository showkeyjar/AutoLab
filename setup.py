from setuptools import setup, find_packages

setup(
    name="autolab",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'requests',
        'pandas',
        'pyyaml'
    ],
)
