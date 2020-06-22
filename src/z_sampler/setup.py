#!/usr/bin/env python

import os

from distutils.core import setup, Extension
import numpy.distutils.misc_util

conda_root = os.environ['CONDA_PREFIX']

extra_dirs = [
    numpy.distutils.misc_util.get_numpy_include_dirs()[0],
    os.path.join(conda_root, 'include'),
    os.path.join(conda_root, 'include', 'gsl'),
    # os.path.join(conda_root, 'Library', 'include'), # OSX extra
    # os.path.join(conda_root, 'Library', 'include', 'gsl'), # OSX extra

]


print("####################################################")
print(conda_root)
print("####################################################")


setup(name             = "zsampler",
      version          = "1.0",
      description      = "Sampling code for the mixture model",
      author           = "Jan Povala",
      author_email     = "jan.povala@gmail.com",
      maintainer       = "jan.povala@gmail.com",
      url              = "jan.povala.com",
      ext_modules      = [
          Extension(
              'zsampler', ['src/z_sampler/src/sampler.c'],
              # extra_compile_args=["-std=c99", "-Ofast", "-march=native", "-I{}".format(os.path.join(conda_root, 'Library', 'include', 'gsl')), "-I{}".format(os.path.join(conda_root, 'include', 'gsl'))],
              # extra_link_args=["-L{}".format(os.path.join(conda_root, 'include', 'gsl')), "-L{}".format(os.path.join(conda_root, 'Library', 'include', 'gsl')), "-lgsl", "-L{}".format(os.path.join(conda_root, 'include')), "-L{}".format(os.path.join(conda_root, 'Library')), "-lcblas"])
              extra_compile_args=["-std=c11", "-O3", "-march=native",  "-I{}".format(os.path.join(conda_root, 'include', 'gsl'))],
              extra_link_args=["-L{}".format(os.path.join(conda_root, 'include', 'gsl')), "-lgsl", "-L{}".format(os.path.join(conda_root, 'include')), "-lcblas"])
      ],
      include_dirs=extra_dirs,
)

