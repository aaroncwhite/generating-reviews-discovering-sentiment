#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = ['pandas',
                'xlrd',
                'numpy', 
                'tensorflow',
                'tqdm',
                'scikit-learn',
                'matplotlib',
                'seaborn', ]

setup_requirements = [
                      'setuptools_scm', 
                      ]

test_requirements = ['pytest', ]

setup(
    author="Aaron White",
    author_email='aaroncwhite@gmail.com',
    classifiers=[
        'Development Status :: Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Package implementation of OpenAI's sentiment neuron",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='sentiment analysis',
    name='sentiment_neuron',
    packages=find_packages(include=['sentiment_neuron']),
    use_scm_version={"root": ".", "relative_to": __file__},
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/aaroncwhite/generating-reviews-discovering-sentiment',
    zip_safe=False,
)