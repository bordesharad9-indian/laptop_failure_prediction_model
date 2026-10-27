from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirement(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    get_requirement = []
    with open(file_path) as file_obj:
        requirement = file_obj.readline()
        requirement = [req.replace("\n","")for req in requirement]

        if HYPHEN_E_DOT in requirement:
            requirement.remove(HYPHEN_E_DOT)

    return requirement

setup(
    name='laptop_failure_prediction',
    version='0.0.1',
    author='sharad',
    author_email='bordesharad9@gmail.com',
    packages=find_packages(),
    install_requires=get_requirement('requirements.txt')
)
