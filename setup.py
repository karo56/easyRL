from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    install_requires = f.read().split("\n")

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="easyRL",
    version="1.0.0",
    author="Karol Muck",
    author_email="karo56.56@gmail.com",
    description="Package to run RL experiments on gym envs with stable-baseline3 library",  # noqa
    long_description=long_description,
    license="MIT",
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
    python_requires=">=3.7",
)
