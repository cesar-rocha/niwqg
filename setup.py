from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='niwqg',
      version='0.1beta',
      description='',
      url='http://github.com/crocha700/niwqg',
      license='MIT',
      packages=['niwqg'],
      install_requires=[
          'numpy',
          'h5py',
      ],
      test_suite = 'nose.collector',
      zip_safe=False)
