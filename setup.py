import setuptools


with open('README_PYPI.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as req_file:
    requirements = [line[:-1] for line in req_file if len(line) > 1]


setuptools.setup(
    name="nn-toolbox",
    version="0.1.3",
    author="Nhat Pham",
    author_email="nphamcs@gmail.com",
    description="A toolbox for common deep learning procedures",
    long_description=long_description,
    url="https://github.com/nhatsmrt/nn-toolbox",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ]
)
