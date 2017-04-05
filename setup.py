#!/usr/bin/env python
"""Python system id.

Includes subspace methods currently.


"""
DOCLINES = __doc__.split("\n")

import os
import sys

from setuptools import setup, find_packages
import versioneer

CLASSIFIERS = """\
Development Status :: 1 - Planning
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Other
Topic :: Software Development
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Scientific/Engineering :: Mathematics
Topic :: Scientific/Engineering :: Physics
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

setup(
    name='sysid',
    maintainer="James Goppert",
    maintainer_email="james.goppert@gmail.com",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    url='https://github.com/jgoppert/sysid',
    author='James Goppert',
    author_email='james.goppert@gmail.com',
    download_url='https://github.com/jgoppert/sysid',
    license='BSD',
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    install_requires=['scipy', 'numpy'],
    tests_require=['nose'],
    test_suite='nose.collector',
    #entry_points = {
        #'console_scripts': ['test=sysid.test:main'],
    #},
    packages=find_packages(
        # choosing to distribute tests
        # exclude=['*.test*']
    ),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
