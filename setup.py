from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

#import cython_gsl

extensions = [
    Extension("pyggop/grbod", ["pyggop/grbod.pyx"]),
    
    Extension("pyggop/fast_flux_computation", ["pyggop/fast_flux_computation.pyx"])
    
]

setup(
    
    name="pyggop",
    
    packages = ["pyggop"],
    
    version = 'v1.0.0',
    
    description = "Generates templates for pair production opacity in relativistic sources",
    
    author = 'Johann Cohen-Tanugi, Giacomo Vianello, Jonathan Granot',
    
    author_email = 'giacomo.vianello@gmail.com',
        
    ext_modules=  cythonize(extensions, 
                            compiler_directives={'cdivision': True}),
    
    
    scripts=['bin/makeggtemplate.py'],
    
    headers=[],
    
    install_requires=['numpy',
                      'scipy',
                      'cython']

)
