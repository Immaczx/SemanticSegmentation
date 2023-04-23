import os
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
   README = readme.read()

os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='semanticSegmentation',
    version='0.0.0',
    packages=['class_activation_maps', 'datasets', 'models', 'losses', 'metrics', 'visualizations'],

    author='Camilo Pelaez Garcia',
    author_email='cpelaezg@unal.edu.co',
    maintainer='Camilo Pelaez Garcia',
    maintainer_email='cpelaezg@unal.edu.co',

    download_url='',

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

    classifiers=[
       'Development Status :: 4 - Beta',
       'Intended Audience :: Developers',
       'Intended Audience :: Education',
       'Intended Audience :: Healthcare Industry',
       'Intended Audience :: Science/Research',
       'License :: OSI Approved :: BSD License',
       'Programming Language :: Python :: 3.7',
       'Programming Language :: Python :: 3.8',
       'Topic :: Scientific/Engineering',
       'Topic :: Scientific/Engineering :: Artificial Intelligence',
       'Topic :: Software Development :: Libraries :: Python Modules',
    ],

)