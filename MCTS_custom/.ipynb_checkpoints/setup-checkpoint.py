from setuptools import setup, find_packages

setup(
    name='monte-carlo-tree-search-custom',
    version=open(".VERSION", "r").read(),
    description='A simple package to allow users to run Monte Carlo Tree Search on any perfect information domain',
    long_description=open("README.md", "r").read(),
    long_description_content_type='text/markdown',
    author='Paul Sinclair, Konstantin Strümpf and others',
    author_email='k.struempf@icloud.com',
    license='MIT',
    url='https://github.com/kstruempf/MCTS',
    keywords=['mcts', 'monte', 'carlo', 'tree', 'search'],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    packages=find_packages()
)
