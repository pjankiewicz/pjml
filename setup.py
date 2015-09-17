from distutils.core import setup

setup(name='pjml',
      version='0.1',
      description='Some ML tools',
      url='http://github.com/pjankiewicz/pjml',
      author='Flying Circus',
      author_email='flyingcircus@example.com',
      license='MIT',
      package_dir={'pjml': 'pjml'},
      packages=['pjml',
                'pjml.sklearn',
                'pjml.sklearn.transformers'],
      zip_safe=False)