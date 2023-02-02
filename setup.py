from setuptools import find_packages, setup

setup(
    name="distillation",
    version="0.0.1",
    description="MSDD",
    packages=find_packages(),
    install_requires=["opencv-python", "SimpleITK", "torchxrayvision","sklearn","matplotlib","timm","blobfile"],
)
