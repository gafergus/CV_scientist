from setuptools import setup, find_packages

with open('requirements.txt') as req_file:
    required = req_file.read().splitlines()

print(required)

setup(
    name='CV_Scientist',
    version='0.09',
    description="A framework for rapidly prototyping CV models",
    author="Glen Ferguson, Michoel Snow, and Tara Blackburn",
    author_email='glen@ferguson76.com',
    packages=find_packages(),
    install_requires=required,
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
)
