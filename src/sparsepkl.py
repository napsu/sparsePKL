'''
Main program for

sparsePKL   - Nonsmooth DC Optimization based Pairwise Kernel Learning Software 
using LMB-DCA and kernels from RLScore (version 0.1)                    
                                                                       
The work was financially supported by the Research Council of Finland 
(Project No. #345804 and #345805).

The sparsePKL software is covered by the MIT license.


First, select the data, loss function, and desired sparsity below. 
Then, run Makefile (by typing "make") to build a shared library that 
allows LMB-DCA (Fortran95 code) to be called from Python program saprsePKL. 
The source uses f2py, Python3.7, and requires a Fortran  compiler 
(gfortran by default) and the RLScore 
 
https://github.com/aatapa/RLScore

to be installed. Finally, just type "python3.7 sparsepkl.py". 


References:

    for sparsePKL:
       N. Karmitsa, K. Joki, "Limited memory bundle DC algorithm for sparse 
       pairwise kernel learning", 2023. 

    for RLScore:
       T. Pahikkala, A. Airola, "Rlscore: Regularized least-squares learners", 
       Journal of Machine Learning Research, Vol. 17, No. 221, pp. 1-5, 2016.

    for LMBM:
       N. Haarala, K. Miettinen, M.M. Mäkelä, "Globally Convergent Limited Memory Bundle Method  
       for Large-Scale Nonsmooth Optimization", Mathematical Programming, Vol. 109, No. 1,
       pp. 181-205, 2007. DOI 10.1007/s10107-006-0728-2.

       M. Haarala, K. Miettinen, M.M. Mäkelä, "New Limited Memory Bundle Method for Large-Scale 
       Nonsmooth Optimization", Optimization Methods and Software, Vol. 19, No. 6, pp. 673-692, 2004. 
       DOI 10.1080/10556780410001689225.

    for NSO:
       A. Bagirov, N. Karmitsa, M.M. Mäkelä, "Introduction to nonsmooth optimization: theory, 
       practice and software", Springer, 2014.

'''
import csv
import itertools as it
import multiprocessing as mp
import numpy as np

from numpy.random import SeedSequence
import time 

from rlscore.kernel import GaussianKernel, LinearKernel
from rlscore.measure import cindex,sqerror

import data         # load pairwise data and divide it to different settings, source copied from A-index with minor modifications.
import sparsepkl    # fortran program
import pkl_utility  # python utility programs for sparsePKL

def run_spkl(params):
    Y = params[0]
    XD = params[1]
    XT = params[2]
    drug_inds = params[3]
    target_inds = params[4]
    training_inds = params[5][0]
    test_inds = params[5][1]
    validation_inds = params[5][2]
    used_data = params[6]
    opt_method = params[7]
    loss = params[8]
    kernels = params[9]
    ireg = params[10]
    autoreg = params[11]
    regparams = params[12]
    nzpros = params[13]
    
    def compute_ka(p,a,n):
        """Calculate prediction p=Ka at the current dual point a."""
        ikro = 0 # For testing different types of Kronecker product computations, use 0 here.

        if ikro == 0: # use vec-trick
            ptmp = pko.matvec(a) 

        else: # compute kronecker product explicitly
            K=pkl_utility.naive_kronecker_kernel(KD, KT, rows1, cols1, rows2, cols2)
            ptmp=np.dot(K,a) 

        for i in range(n): # This will not change in the calling subroutine unless changed elementwise.
            p[i]=ptmp[i]

        return

    """Reprocessing data"""
    # Indices for training, validation, and test    
    train_drug_inds = drug_inds[training_inds]
    train_target_inds = target_inds[training_inds]
    Y_train = Y[training_inds]

    test_drug_inds = drug_inds[test_inds]
    test_target_inds = target_inds[test_inds]
    Y_test = Y[test_inds]
    
    validation_drug_inds = drug_inds[validation_inds]
    validation_target_inds = target_inds[validation_inds]
    Y_validation = Y[validation_inds]

    # The used setting S1-S4 is not explicitly given so it needs to be found out.
    if set(test_drug_inds).isdisjoint(set(train_drug_inds)):
        if set(test_target_inds).isdisjoint(set(train_target_inds)):
            setting = "S4"
        else:
            setting = "S3"
    else:
        if set(test_target_inds).isdisjoint(set(train_target_inds)):
            setting = "S2"
        else:
            setting = "S1"

    Y = np.array(Y_train, dtype='d', order='F')
    nrec = len(Y) 

    """ Defining kernels """
    # Compute kernel matrices for drugs and targets
    drug_kernel_type = kernels[0][0]
    if drug_kernel_type == "linear":
        drug_kernel = LinearKernel(XD)
    elif drug_kernel_type == "gaussian":
        drug_kernel = GaussianKernel(XD, gamma=10**-5) # -5 is default
    KD = drug_kernel.getKM(XD)

    target_kernel_type = kernels[0][1]
    if target_kernel_type == "linear":
        target_kernel = LinearKernel(XT)
    elif target_kernel_type == "gaussian":
        target_kernel = GaussianKernel(XT, gamma=10**-5) # -5 is default
    KT = target_kernel.getKM(XT)

    # Create training, validation and test kernels (separate).  No retraining after validation.
    KD_train, rows1, rows2 = pkl_utility.K_to_dense(KD, train_drug_inds, train_drug_inds)
    KT_train, cols1, cols2 = pkl_utility.K_to_dense(KT, train_target_inds, train_target_inds)
    KD_val, rows_val1, rows_val2 = pkl_utility.K_to_dense(KD, validation_drug_inds, train_drug_inds)
    KT_val, cols_val1, cols_val2 = pkl_utility.K_to_dense(KT, validation_target_inds, train_target_inds)
    KD_test, rows_test1, rows_test2 = pkl_utility.K_to_dense(KD, test_drug_inds, train_drug_inds)
    KT_test, cols_test1, cols_test2 = pkl_utility.K_to_dense(KT, test_target_inds, train_target_inds)

    # Pairwise kernels
    pko_function = kernels[0][2]
    pko = eval('pkl_utility.'+pko_function+'(KD_train, KT_train, rows1, cols1, rows2, cols2)')
    pkoval =  eval('pkl_utility.'+pko_function+'(KD_val, KT_val, rows_val1, cols_val1, rows_val2, cols_val2)')     
    pkotest = eval('pkl_utility.'+pko_function+'(KD_test, KT_test, rows_test1, cols_test1, rows_test2, cols_test2)')
    

    """ Parameters for optimization """ 
    # Selection of the method
    if opt_method == "LMB-DCA":
        sparsepkl.initlmbdca.imet = 0
    elif opt_method == "DCA":
        sparsepkl.initlmbdca.imet = 1
    else:
        print('Sorry, no optimization method '+opt_method+' coded.')
    
    m = len(rows1)
    n = len(rows2)

    iterm=np.array(0,dtype=np.int32) # This needs to be an array in order to change it in fortran
    if autoreg == 1:
        sparsepkl.initpkl.autolambda = 1
    else:
        sparsepkl.initpkl.autolambda = 0
        sparsepkl.initpkl.rho = np.float32(regparams[0]) 
        sparsepkl.initpkl.rho2 = sparsepkl.initpkl.rho
    rhoupdate = 30.0    
    sparsepkl.initpkl.ireg = ireg
    
    if nzpros < 1.0:
        sparsepkl.initpkl.k = int(0.99*nzpros*nrec) # the number of elements we allow to be nonzero.
    else:
        sparsepkl.initpkl.k = int(nzpros*nrec) # no need for safequard if a sparse solution is not wanted. 
    
    CIbestvali = 0.0
    CIbesttest = 0.0
    MSEbestvali = 1.7976931348623157e+308
    MSEbesttest = 1.7976931348623157e+308
    best_lam_CI = 0.0
    best_lam_MSE = 0.0
    
    nzprev=nrec+1
    zero_limit = 1e-4 

    # Starting time
    usedtime0 = time.process_time()

    # Starting point
    apy = np.ones(int(nrec), dtype='d', order='F')/nrec # initialization of dual variable for Fortran

    MSEbestvali = 1.7976931348623157e+308
    MSEbesttest = 1.7976931348623157e+308
    CIbestvali = 0.0
    CIbesttest = 0.0
    itbestMSE = 0
    itbestCI = 0
    for h in range(50): 
    #for h in range(1): # when testing only optimization 

        sparsepkl.fmodule.sparsepkl(compute_ka,apy,Y,loss,iterm,nrec,sparsepkl.initpkl.k)
        P_val = pkoval.matvec(apy)
        P_test = pkotest.matvec(apy)

        # Compute C-index
        valiCI = cindex(Y_validation, P_val)
        testCI = cindex(Y_test, P_test)
        print("C-index in validation data for %s with %s after %i iterations with setting %s:  %f" %(opt_method, loss, int((h+1)*500), setting, valiCI))
        print("C-index in test data for %s with %s after %i iterations with setting %s:        %f" %(opt_method, loss, int((h+1)*500), setting, testCI))
        
        # Compute MSE
        valiMSE = sqerror(Y_validation, P_val)
        testMSE = sqerror(Y_test, P_test)
        #print("MSE in validation data for %s with %s after %i iterations with setting %s: %f" %(opt_method, loss, int((h+1)*500), setting, valiMSE))
        #print("MSE in test data for %s with %s after %i iterations with setting %s:       %f" %(opt_method, loss, int((h+1)*500), setting, testMSE))

        # Additional performance indices
        # If you want to compute group performance C-indices w.r.t. drugs and targets uncomment the following lines
        #gp1 = pkl_utility.group_performance(cindex, Y_test, P_test, test_drug_inds)
        #print("C-index w.r.t. drugs: ",gp1)
        #gp2 = pkl_utility.group_performance(cindex, Y_test, P_test, test_target_inds)
        #print("C-index w.r.t. targets",gp2)

        # Sparse solution
        nz = len(apy[np.abs(apy) > zero_limit])
        for i in range(n): # Tätä kannattaa testata kannattaako laittaa
            if np.abs(apy[i]) <= zero_limit:
                apy[i] = 0.0 

        P_test_sparse = pkotest.matvec(apy)
        testCIsparse = cindex(Y_test, P_test_sparse)
        print("Sparse test set C-index for %s with %s after %i iterations with setting %s:     %f" %(opt_method, loss, int((h+1)*500), setting, testCIsparse))
        testMSEsparse = sqerror(Y_test, P_test_sparse)
        #print("Sparse test set MSE for %s after %i iterations with setting %s:     %f" %(opt_method, int((h+1)*100), setting, testMSEsparse))
        print("Number of nonzero elements in a     ",nz)
        print("Number of elements in a             ",n)
        print("Percentage of nonzero elements in a ",100.0*nz/n)
        # end sparse solution
            
        usedtime = time.process_time() - usedtime0
        if nz <= nzpros*nrec: # Desired sparsity
            if CIbestvali < valiCI:
                CIbestvali = valiCI
                CIbesttest = testCIsparse
                MSEbestwithCI = testMSEsparse
                itbestCI = int((h+1)*500)
                nzCI = 100.0*nz/n
                best_lam_CI = np.float32(sparsepkl.initpkl.rho)
                PCI = P_test_sparse

            if MSEbestvali > valiMSE:
                MSEbestvali = valiMSE
                MSEbesttest = testMSEsparse
                CIbestwithMSE = testCIsparse
                itbestMSE = int((h+1)*500)
                nzMSE = 100.0*nz/n
                best_lam_MSE = np.float32(sparsepkl.initpkl.rho)
                PMSE = P_test_sparse

            if (iterm == 1 or iterm == 2 or iterm == 9): 
                print ("Optimization successfully terminated!\n")
                print('CPU time = %4.2f' %usedtime)
                print("The final solution w.r.t. C-index and MSE with %s applying %s after %i iterations with setting %s: %f and %f" %(opt_method, loss, int((h+1)*500), setting, CIbesttest, MSEbesttest))
                print('\n')
                if nzpros < 1.0 or h>4: # to avoid premature termination in cases k=n
                    break

        else: # Not yet enough sparsity optained
            if nz < nzprev: # Save best so far solution
                CIbesttest = testCIsparse
                MSEbestwithCI = testMSEsparse
                itbestCI = int((h+1)*500)
                nzCI = 100.0*nz/n
                best_lam_CI = np.float32(sparsepkl.initpkl.rho)
                PCI = P_test_sparse
                MSEbesttest = testMSEsparse
                CIbestwithMSE = testCIsparse
                itbestMSE = int((h+1)*500)
                nzMSE = 100.0*nz/n
                best_lam_MSE = np.float32(sparsepkl.initpkl.rho)
                PMSE = P_test_sparse

            print("Not yet enough sparsity optained. Updating lambda to ",rhoupdate*sparsepkl.initpkl.rho,rhoupdate)
            sparsepkl.initpkl.rho = rhoupdate*sparsepkl.initpkl.rho
            rhoupdate = max(1.20,rhoupdate/2.0) 

        # Print the intermediate result
        print('CPU time = %4.2f' %usedtime)
        print("The best solution so far w.r.t. C-index and MSE with %s applying %s after %i iterations with setting %s: %f and %f" %(opt_method, loss, int((h+1)*500), setting, CIbesttest, MSEbesttest))
        print('\n')

    # Save predictions
    field = ["Setting","Y_test", "P_test (CI)", "P_test (MSE)"]
    with open('sparsePKL_predictions_'+opt_method+'_'+loss+'_'+ds+'_'+setting+'_'+str(random_seed)+'.csv', 'w') as f:
        writer = csv.writer(f, delimiter=";", lineterminator="\n")
        writer.writerow(field)
        for i in range(len(Y_test)):
            predi = [setting,Y_test[i],PCI[i],PMSE[i]]
            writer.writerow(predi)

    return(setting,nzpros,best_lam_CI,CIbesttest,MSEbestwithCI,nzCI,itbestCI,best_lam_MSE,CIbestwithMSE,MSEbesttest,nzMSE,itbestMSE,usedtime) 

if __name__ == "__main__":
    base_seed = 12345
    repetitions = 5
    ss = SeedSequence(base_seed)
    random_seeds = ss.generate_state(repetitions)
    
    # Select the data from the list below (or add your own with appropriate loading procedure)
    #datasets = ["davis"]
    #datasets = ["metz"]
    #datasets = ["kiba"]
    #datasets = ["merget"]
    #datasets = ["GPCR"]
    #datasets = ["IC"]
    #datasets = ["E"]
    datasets = ["kiba","merget","E"]
    #datasets = ["davis","metz","kiba","merget","GPCR","IC","E"]
    
    # Select percentage of samples in training data with setting S1
    split_percentage = 1.0/3

    # Select the optimization method from the list below (only one at the time!)
    opt_methods = "LMB-DCA" # (default)
    #opt_methods = "DCA" 
    
    # Select the loss function from the list below (only one at the time!)
    #loss = "RLS"
    #loss = "L1"
    #loss = "hinge-loss" 
    #loss = "semi-squared-hinge" 
    loss = "squared-hinge" 
    #loss = "svm-hinge" # the results are not convincing

    # Select the kernels (KD, KT, K_pairwise) from the list below (only one combination at the time).
    kernels = [["gaussian", "gaussian", "pko_kronecker"]]
    #kernels = [["linear", "linear", "pko_linear"]] # the results are not convincing
    #kernels = [["linear", "linear", "pko_kronecker"]] # the results are not convincing

    # Select regularization
    ireg = 1 # Switch for regularization: 0 = only L0-norm, 1 = double regularization with L1- and L0-norms (default). 
    autoreg = 1 # Type of regularization: 0 = selected first regularization parameter (give the value below), 1 = automatic regularization (default).
    regparam = [0.0001] # Used with autoreg == 0.
    
    # Part of elements in dual vector we allow to be nonzero (i.e. 0.5 is 50%). 
    # Note, if you select nz-percentage = 1.0, the code still computes k-norm. I.e. lots of extra computation is made.
    #nz_percentage = [0.50] 
    nz_percentage = [1.0,0.50,0.20,0.10] 
    
    for ds in datasets:
        
        # Load data
        XD, XT, Y, drug_inds, target_inds = eval('data.load_'+ds+'()')    
        n_D = XD.shape[0]
        n_T = XT.shape[0]

        print('\n')
        print('sparsePKL: optimization method:    ',opt_methods)
        print('           loss function:          ',loss)
        print('           used data:              ',ds, 'with',n_D,'drugs and',n_T,'targets.')
        print('           kernel for drugs:       ',kernels[0][0])
        print('           kernel for targets:     ',kernels[0][1])
        print('           pairwise kernel:        ',kernels[0][2])
        print('\n')

        for random_seed in random_seeds:
            #if random_seed == 2688385916 or random_seed == 3048105090 or random_seed == 4196366895:
            if random_seed == 1:
                pass
            else:
                field = ["Setting","nz%","lambda (CI)", "C-index (CI)", "MSE (CI)","nz% (CI)","nit (CI)","lambda (MSE)", "C-index (MSE)", "MSE (MSE)","nz% (MSE)","nit (MSE)","CPU-time"]
                with open('SPKL6_indices_'+opt_methods+'_'+loss+'_'+ds+'_'+str(random_seed)+'.csv', 'w') as f:
                    writer = csv.writer(f, delimiter=";", lineterminator="\n")
                    writer.writerow(field)
        
                for nz_p in nz_percentage:

                    time_start = time.time()
        
                # Create the splits for the four different experimental settings such that there is one common test set 
                # for every setting. The split_percentage defines number of training samples in S1 setting. The rest of 
                # samples are divided to test and validation.
                    df, splits = data.splits(drug_inds, target_inds, split_percentage, random_seed)
                    print("Splits in "+ds+" calculated in time", time.time()-time_start)
                    splits_1234 = list(it.chain.from_iterable(splits))
            
                    parameters = it.product([Y], [XD], [XT], [drug_inds], [target_inds], splits_1234,[ds],[opt_methods],[loss],[kernels],[ireg],[autoreg],[regparam],[nz_p])
        
                # Compute different settings at the same time.
                    pool = mp.Pool(processes = 4)
                    output = pool.map(run_spkl, list(parameters))
                    pool.close()
                    pool.join()

                    print(output)
                    print('\n')
                
                # Save result indices (predictions are saved above)
                    print("printing results 1")
                    with open('SPKL6_indices_'+opt_methods+'_'+loss+'_'+ds+'_'+str(random_seed)+'.csv', 'a') as f:
                        print("printing results 2")
                        writer = csv.writer(f, delimiter=";", lineterminator="\n")
                        print("printing results 3")
                        writer.writerows(output)
                    print("printing results 4")
                

