from setuptools import setup, find_packages

from setuptools.extension import Extension
#from Cython.Build import cythonize

#read more on : 
#http://python-packaging.readthedocs.io/en/latest/minimal.html
#for projects not from pypi: dependency_links = ['url']
#

"""
extensions = [
    Extension("rdmresearch_boost.crdmresearch",["rdmresearch_boost/crdmresearch.pyx"],
        include_dirs=[],
        libraries=[],
        library_dirs=[]),
    Extension("rdmresearch_boost.ctest",["rdmresearch_boost/ctest.pyx","rdmresearch_boost/ctest_c.cpp","rdmresearch_boost/mat.cpp"],
        language="c++"),
    Extension("rdmresearch_boost.pyxinit",["rdmresearch_boost/pyxinit.pyx"]),
    Extension("rdmresearch_boost.parallel",["rdmresearch_boost/parallel.pyx"]),
    Extension("rdmresearch_boost.rdmcoordinates",
        ["rdmresearch_boost/rdmcoordinates.pyx"]),
    Extension("rdmresearch_boost.rdmlocators",
        ["rdmresearch_boost/rdmlocators.pyx"]),
    Extension("rdmresearch_boost.rdmtrace",
        ["rdmresearch_boost/rdmtrace.pyx"]),  
    Extension("rdmresearch_boost.service",
        ["rdmresearch_boost/service.pyx"]),
]
"""


setup(name='st_keras',
        #packages=['rdmresearch'],
        packages=find_packages(),
        #ext_modules = cythonize(extensions),
        install_requires=[
            'st_core'
        ],
        version='0.0.0.dev1')
