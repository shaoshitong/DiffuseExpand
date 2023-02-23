from setuptools import find_packages, setup

setup(
    name="diffuseexpand",
    version="0.0.1",
    description="Segmentation Dataset Expansion",
    packages=find_packages(),
    install_requires=["opencv-python", "SimpleITK", "torchxrayvision","sklearn","matplotlib","timm","blobfile"],
)
