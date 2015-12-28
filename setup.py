from setuptools import setup, find_packages
from pip.req import parse_requirements
from pip.download import PipSession


requires = [str(i.req) for i in parse_requirements('requirements.txt',
                                                   session=PipSession())
            if i.req is not None]


setup(name='ramjet',
      version='1.1.1',
      author="Laisky",
      author_email="ppcelery@gmail.com",
      description="Timing tasks framework.",
      license="MIT/Apache",
      keywords="timing tasks frameworks asynchrous",
      packages=find_packages('src'),
      package_dir={'': 'src'},
      include_package_data=True,
      install_requires=requires,
      classifiers=[
          "Development Status :: 4 - Alpha",
          'Topic :: Software Development :: Libraries',
          "Environment :: Web Environment",
          "Intended Audience :: Developers",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          'Programming Language :: Python :: 3.4',
          "Framework :: Tornado",
      ],
      zip_safe=True)
