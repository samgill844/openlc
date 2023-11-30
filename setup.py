import warnings
#warnings.filterwarnings("ignore")

from setuptools import setup, Extension, find_packages
#from numpy.distutils.core import setup

    
setup(
    name = 'openlc',
    version = '0.1',
    description = 'GPU-accelerated binary star model',
    url = None,
    author = 'Samuel Gill et al',
    author_email = 'samuel.gill@warwick.ac.uk',
    license = 'GNU',
    #packages=['bruce','pwdbruce/binarystar'],
    packages = find_packages(),
    package_data={'openlc': ['c_src/total.c']},
    scripts=['programs/lcmatch'],
    install_requires=['pyopencl'] , zip_safe=False,)