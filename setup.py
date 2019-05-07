from setuptools import setup, find_packages

with open('requirements.txt') as req_file:
    required = req_file.read().splitlines()

print(required)

setup(
    name='CV_Scientist',
    version='0.01',
    description="Aframework for rapidly prototyping CV model using keras and TF",
    author="Glen Ferguson and Michoel Snow",
    author_email='glen@ferguson76.com',
    packages=find_packages(),
    install_requires=required,
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
)
