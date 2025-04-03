# sparsePKL - Sparse Pairwise Kernel Learning Software 

__sparsePKL__ is a pairwise kernel learning algorithm based on nonsmooth DC (difference of two convex functions) optimization. It learns sparse models for predicting in pairwise data (e.g. drug-target interactions) by using double regularization with both L1-norm and L0-pseudonorm. The nonsmooth DC optimization problem is solved using the limited memory bundle DC algorithm (LMB-DCA).  In addition, __sparsePKL__ uses pairwise Kronecker product kernels computed via generalized vec-trick to model interactions between drug and target features. The included loss-functions for the pairwise kernel problem are:
* squared loss,
* squared epsilon-insensitive loss,
* epsilon-insensitive squared loss,
* epsilon-insensitive absolute loss,
* absolute loss.



## Files included
* sparsepkl.py
  - Main python file. Includes [RLScore](https://github.com/aatapa/RLScore) calls.
* pkl_utility.py
  - Python utility programs.
* sparsepkl.f95
  - Main Fortran file for sparsePKL software.
* lmbdca.f95
  - LMB-DCA - the limited memory bundle DC algorithm.
* solvedca.f95
  - Limited memory bundle method for solving convex DCA-type of problems.
* objfun.f95
  - Computation of the function and subgradients values with different loss functions. Selection between loss functions is made in sparsepkl.py
* initpkl.f95
  - Initialization of parameters and variables in sparsePKL and LMB-DCA. Includes modules:
    + initpkl     - Initialization of parameters for pairwise learning.
    + initlmbdca  - Initialization of LMB-DCA.
* parameters.f95
  - Parameters for Fortran. Inludes modules:
    + r_precision - Precision for reals,
    + param - Parameters,
    + exe_time - Execution time.
* subpro.f95
  - subprograms for LMB-DCA and LMBM.
* data.py
  - Contains functions to load the example data sets. Data files are assumed to be in a folder "data" that is not part of the current folder.
  - Contains functions to create train-test-validation splits. Splits are created for every experimental setting S1-S4 (see the reference below).

* Makefile
  - makefile: builds a shared library to allow sparsepkl (Fortran95 code) to be called from Python. Uses f2py, Python3.7, and requires a Fortran compiler (gfortran) to be installed.


## Installation and usage
The source uses f2py and Python3.7, and requires a Fortran  compiler (gfortran by default) and the [RLScore](https://github.com/aatapa/RLScore) to be installed.

To use the code:
1) Select the data, loss function, and the desired sparsity level from sparsepkl.py file.
2) Run Makefile (by typing "make") to build a shared library that allows sparsepkl (Fortran95 code) to be called from Python. 
3) Finally, just type "python3.7 sparsepkl.py".

The algorithm returns a csv-file with performance measures (C-index and MSE) computed in the test set under different experimental settings S1-S4. The best results are selected using a separate validation set and validated w.r.t. C-index.
In addition, separate csv-files with predictions under different experimental settings S1-S4 are returned. 

## References:

* sparsePKL and LMB-DCA:
  - N. Karmitsa, K. Joki, A. Airola, T. Pahikkala, "[Limited memory bundle DC algorithm for sparse pairwise kernel learning](https://link.springer.com/article/10.1007/s10898-025-01481-w?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=oa_20250403&utm_content=10.1007/s10898-025-01481-w)", _Journal of Global Optimization_, 2025.  
* [RLScore](https://github.com/aatapa/RLScore):
  - T. Pahikkala, A. Airola, "[Rlscore: Regularized least-squares learners](https://www.jmlr.org/papers/volume17/16-470/16-470.pdf)", _Journal of Machine Learning Research_, Vol. 17, No. 221, pp. 1-5, 2016.
* LMBM:
  - N. Haarala, K. Miettinen, M.M. Mäkelä, "[Globally Convergent Limited Memory Bundle Method for Large-Scale Nonsmooth Optimization](https://link.springer.com/article/10.1007/s10107-006-0728-2)", _Mathematical Programming_, Vol. 109, No. 1, pp. 181-205, 2007.
  - M. Haarala, K. Miettinen, M.M. Mäkelä, "[New Limited Memory Bundle Method for Large-Scale Nonsmooth Optimization](https://www.tandfonline.com/doi/abs/10.1080/10556780410001689225)", _Optimization Methods and Software_, Vol. 19, No. 6, pp. 673-692, 2004.
* Generalized vec trick and experimental settings:
  - A. Airola, T. Pahikkala, "[Fast kronecker product kernel methods via generalized vec trick](https://ieeexplore.ieee.org/document/7999226)", _IEEE Transactions on Neural Networks and Learning Systems_, Vol. 29, pp. 3374–3387, 2018.
  - M. Viljanen, A. Airola, T. Pahikkala, "[Generalized vec trick for fast learning of pairwise kernel models](https://link.springer.com/article/10.1007/s10994-021-06127-y)", _Machine Learning_, Vol. 111, 543–573, 2022.
* Nonsmooth optimization:
  - A. Bagirov, N. Karmitsa, M.M. Mäkelä, "[Introduction to nonsmooth optimization: theory, practice and software](https://link.springer.com/book/10.1007/978-3-319-08114-4)", _Springer_, 2014.

## Acknowledgements
The work was financially supported by the Research Council of Finland projects (Project No. #345804 and #345805) led by Antti Airola and Tapio Pahikkala.


   
