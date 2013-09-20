pynnfbp
=======

Python implementation of the NN-FBP Algorithm, published in [1].

To install PyNN-FBP, simply run:

`python setup.py install`

To use PyNN-FBP, you need installed:

- Numpy and scipy
- PyTables
- (for ASTRAProjector) The ASTRA toolbox (https://code.google.com/p/astra-toolbox/), with Python interface (https://github.com/dmpelt/pyastratoolbox)
- (for SimplePyCUDAProjector) PyCUDA and pyfft.

Examples can be found in the `examples` directory. After installation, the examples can be run to show how the package is used.

Running `PaperExample.py` from the `examples` directory should show comparable results to the threeshape experiment (Fig. 9a and Table 1) of [1].

[1] Fast tomographic reconstruction from limited data using artificial neural networks, Daniel M. Pelt and Kees Joost Batenburg, *submitted for publication*, 2013
