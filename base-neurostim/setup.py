from setuptools import setup

setup(
    name='neurostim',
    version='0.1.0',    
    description='Simulate neuronal responses to optogenetic stimulation',
    url='https://github.com/dberling/simneurostim',
    author='David Berling',
    author_email='berling@ksvi.mff.cuni.cz',
    license='GPL-3.0',
    packages=['neurostim'],
    install_requires=[],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License ::GPL-3.0 ',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
    ],
)
