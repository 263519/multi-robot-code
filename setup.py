from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'project1'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='danjayy',
    maintainer_email='lifteddebbymartha@gmail.com',
    description='Package implementing the first part of the MRS Project 1',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'basic_rules = project1.basic_rules:main'
        ],
    },
)
