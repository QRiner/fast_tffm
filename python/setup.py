import sys
from setuptools import setup, find_packages

version = 0.1

if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported')

setup(
    name='fast_tffm',
    version=version,
    description='fast_tffm',
    packages=find_packages(exclude=('tests', 'tests.*')),
    include_package_data=True,
    entry_points={
        'console_scripts': ['fast_tffm = fast_tffm.cmdline:main']
    },
    package_data={
        'fast_tffm': ['../lib/*.so'],
    },
    install_requires=[
        'tensorflow>=1.13.1',
    ]
)
