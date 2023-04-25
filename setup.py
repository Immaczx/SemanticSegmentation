import os
from setuptools import setup,find_packages

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
   README = readme.read()

os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='SemanticSegmentation',
    version='0.0.0',
    packages=find_packages(
        where='src',
        include=['losses*'],  # alternatively: `exclude=['additional*']`
    ),
    package_dir={"SemanticSegmentation": "src"},
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

    long_description=README,
    long_description_content_type='text/markdown',

)