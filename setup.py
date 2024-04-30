from typing import List
from setuptools import setup,find_packages
hypen='-e .'
def get_requirement(file_path:str)->List[str]:

    requirements=[]

    with open (file_path) as file_obj:
        requirement=file_obj.readlines()
        for line in requirement:
            requirement=line.strip()
            if requirement.startswith('/'):
                requirement=requirement[1:]
            if requirement != hypen:
                requirements.append(requirement)

    return requirements



setup(
    name='ml project',
    authore='sk muzahid',
    author_email='muzahidsk771@gmail.com',
    packages=find_packages(),
    install_requires=get_requirement('requirements.txt')
)