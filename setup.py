from pathlib import Path

from setuptools import setup

# create the hidden folder that holds the editable utils install
Path(".utils.egg-info").mkdir(exist_ok=True)

# install the utils package
setup(
    name="utils",
    package_dir={"": "src/utils"},
)
