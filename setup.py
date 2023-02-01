from setuptools import find_packages,setup
from typing import List

requirements_file_name="requirements.txt"
REMOVE_PACKAGE="-e ."

def get_requirements()->List[str]:
    with open(requirements_file_name) as f:
        requirement_list = f.readline()
    requirement_list = [requirement_name.replace("\n","") for requirement_name in requirement_list]

    if REMOVE_PACKAGE in requirement_list:
        requirement_list.remove(REMOVE_PACKAGE)
        
    return requirement_list



setup(name='Insurance',
      version='0.0.1',
      description='Insurance Industry Level Project',
      author='Abhineash Kishan',
      author_email='avikishan258@gmail.com',
      packages=find_packages(),
      install_require=get_requirements()
    )