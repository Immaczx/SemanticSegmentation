import os
from setuptools import setup, package_index

setup(
    name='SemanticSegmentation',
    version='0.0.0',
    packages= package_index(),
    author='Camilo Pelaez Garcia',
    author_email='cpelaezg@unal.edu.co',
    install_requires=['scikit-image',
                     'matplotlib',
                     'gdown',
                     'opencv-python',
                     'scikit-learn'  
    ],
    include_package_data=True,
    license='Simplified BSD License',
    description="",
    zip_safe=False,
    python_requires='>=3.6',
)