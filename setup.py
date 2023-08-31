from setuptools import setup

setup(
    name='BiG',
    version='0.1.0',
    description=' Code for measuring 3-D bispectra from density maps using GPUs',
    url='',
    author='Laila Linke',
    author_email='laila.linke@uibk.ac.at',
    packages=['BiG'],
    install_requires=['numpy',
                      'argparse',
                      'os',
                      'pathlib',
                      'nbodykit',
                      'jax'],
    classifiers=[
        'Development Status :: 2 - PreAlpha',
        'Environment :: GPU :: NVIDIA CUDA :: 12 :: 12.1'
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],

)