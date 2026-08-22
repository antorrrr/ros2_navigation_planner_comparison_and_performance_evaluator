from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'nav2_performance_evaluator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*_launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='antor',
    maintainer_email='antor@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'evaluator = nav2_performance_evaluator.evaluator:main',
            'param_ev = nav2_performance_evaluator.param_ev:main',
        ],
    },
)
