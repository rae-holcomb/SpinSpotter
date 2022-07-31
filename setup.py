from setuptools import setup, find_packages

setup(
    name='spinspotter',
    version='0.1.3',    
    description='Fast, autocorrelation-based algorithm for finding stellar rotation periods from light curves.',
    url='https://github.com/rae-holcomb/SpinSpotter',
    author='Rae Holcomb',
    author_email='raeholcomb15@gmail.com',
    license='BSD 3-clause',
    packages=find_packages(),
    install_requires=['matplotlib',
                      'numpy',  
                      'pandas',
                      'scipy',
                      'statsmodels',
                      'lightkurve>=2.0'                   
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
)
