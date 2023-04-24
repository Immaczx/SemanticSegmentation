import os
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
   README = readme.read()

os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='semanticSegmentation',
    version='0.0.0',
    packages=['class_activation_maps', 'datasets.segmentation', 'models', 'losses', 'metrics', 'visualizations', 'utils'],
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
    long_description=README,
    long_description_content_type='text/markdown',
    python_requires='>=3.6',
)