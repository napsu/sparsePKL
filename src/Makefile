#*************************************************************************
#*                                                                       *
#*     Makefile for sparsePKL --- A Python combined nonsmooth DC         *
#*     optimization based pairwise kernel learning code                  *
#*                                                                       *
#*************************************************************************
#
# Build a shared library to allow LMB-DCA (Fortran95 code) to be
# called from Python program sparsePKL. Uses f2py, Python3.7, and requires 
# a Fortran compiler (here gfortran) to be installed.
#
# After running this makefile, run the program by typing 
#   python3.7 sparsepkl.py
# 
#FF = gfortran -O2 -ffast-math
EXT_SUFFIX := $(zsh python3-config --extension-suffix)
FF = gfortran -O2
PF = python3.7 -m numpy.f2py
FLAGS = 
#FLAGS = --f90flags='-O2'
#FLAGS = --fcompiler=gfortran --f90flags='-fimplicit-none -O2'

all:	sparsepkl$(EXT_SUFFIX)
#
# ---- Compile Fortran files --------------------------- 
parameters.o: parameters.f95
	$(FF) -c parameters.f95

initpkl.o: r_precision.mod initpkl.f95
	$(FF) -c initpkl.f95

objfun.o: r_precision.mod objfun.f95
	$(FF) -c objfun.f95

subpro.o: r_precision.mod param.mod subpro.f95
	$(FF) -c subpro.f95

solvedca.o: r_precision.mod param.mod initpkl.mod initlmbdca.mod exe_time.mod obj_fun.mod subpro.mod solvedca.f95
	$(FF) -c solvedca.f95

lmbdca.o: r_precision.mod param.mod initpkl.mod initlmbdca.mod exe_time.mod obj_fun.mod subpro.mod lmbdca.f95
	$(FF) -c lmbdca.f95

# ---- Create Signature ------------------
sparsepkl.pyf: parameters.o initpkl.o objfun.o subpro.o solvedca.o lmbdca.o sparsepkl.f95
	$(PF) --overwrite-signature parameters.f95 initpkl.f95 objfun.f95 subpro.f95 solvedca.f95 lmbdca.f95 sparsepkl.f95 -m sparsepkl -h sparsepkl.pyf

# ---- Compile and Link --------------------------- 
sparsepkl$(EXT_SUFFIX): parameters.o initpkl.o objfun.o subpro.o solvedca.o lmbdca.o sparsepkl.f95 sparsepkl.pyf
#hellocall.so: parameters.o hellocall2.f95 hellocall.f95 hellocall.pyf
	$(PF) -c $(FLAGS) sparsepkl.pyf parameters.f95 initpkl.f95 objfun.f95 subpro.f95 solvedca.f95 lmbdca.f95 sparsepkl.f95

# ---- Clean -----------------------------
clean:	
	rm -rf *.o *.mod *.so *.pyf __pycache__
	echo Clean done	