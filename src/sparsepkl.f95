!*************************************************************************
!*                                                                       *
!*     sparsePKL - Nonsmooth DC Optimization based Pairwise Learning     *
!*                 Software using LMB-DCA and kernels from RLScore       *
!*                 (version 0.1)                                         *
!*                                                                       *
!*     by Napsu Karmitsa 2023 (last modified 31.08.2023).                *
!*                                                                       *
!*     The sparsePKL software is covered by the MIT license.             *
!*                                                                       *
!*************************************************************************
!*
!*
!*     Codes included:
!*
!*     sparsepkl.py          - Main Python file. Includes RLScore calls.
!*     pkl_utility.py        - Python utility programs.
!*     sparsepkl.f95         - Building plock between Python and Fortran for 
!*                             pairwise learning software (this file).
!*     parameters.f95        - Parameters. Inludes modules:
!*                               - r_precision - Precision for reals,
!*                               - param - Parameters,
!*                               - exe_time - Execution time.
!*     initpkl.f95           - initialization of PKL parameters and LMB-DCA.
!*                             Includes modules:
!*                               - initpkl     - Initialization of parameters for learning.
!*                               - initlmbdca  - Initialization of LMB-DCA.
!*     lmbdca.f95            - LMB-DCA - limited memory bundle algorithm for DC optimization.
!*     solvedca.f95          - limited memory bundle method for solving pure DCA problem.
!*     objfun.f95            - computation of the function and subgradients values
!*     subpro.f95            - subprograms for LMB-DCA.
!*     data.py               - reading and splitting of example data sets.
!*
!*     Makefile              - makefile: builds a shared library to allow LMB-DCA (Fortran95 code)
!*                             to be called from Python program sparsePKL. Uses f2py, Python3.7, 
!*                             and requires a Fortran compiler (gfortran) to be installed.
!*
!*
!*
!*    After running the makefile (type "make"), run the program by typing
!*
!*      python3.7 sparsepkl.py
!*
!*
!*    To change the parameter of the optimization software modify initpkl.f95 
!*    as needed. If you do, rerun Makefile.
!*
!*
!*     References:
!*
!*     for sparsePKL:
!*
!*
!*     for RLScore:
!*
!*       https://github.com/aatapa/RLScore
!*
!*       T. Pahikkala, A. Airola, "Rlscore: Regularized least-squares learners", 
!*       Journal of Machine Learning Research, Vol. 17, No. 221, pp. 1-5, 2016.
!*
!*     for LMBM (used as underlying solver for DCA):
!*       N. Haarala, K. Miettinen, M.M. Mäkelä, "Globally Convergent Limited Memory Bundle Method  
!*       for Large-Scale Nonsmooth Optimization", Mathematical Programming, Vol. 109, No. 1,
!*       pp. 181-205, 2007. DOI 10.1007/s10107-006-0728-2.
!*
!*       M. Haarala, K. Miettinen, M.M. Mäkelä, "New Limited Memory Bundle Method for Large-Scale 
!*       Nonsmooth Optimization", Optimization Methods and Software, Vol. 19, No. 6, pp. 673-692, 2004. 
!*       DOI 10.1080/10556780410001689225.
!*
!*     for NSO:
!*       A. Bagirov, N. Karmitsa, M.M. Mäkelä, "Introduction to nonsmooth optimization: theory, 
!*       practice and software", Springer, 2014.
!*
!*
!*     Acknowledgements:
!*
!*     The work was financially supported by the Research Council of Finland projects (Project No. #345804
!*     and #345805) led by Antti Airola and Tapio Pahikkala.
!*
!*************************************************************************
!*
!*     * fmodule sparsepkl *
!*
!*     Main Fortran program for pairwise learning software with LMB-DCA.
!*
!*************************************************************************

MODULE fmodule

    USE r_precision, ONLY : prec  ! Precision for reals.
    USE lmbdca
    USE obj_fun
    IMPLICIT NONE

CONTAINS
    SUBROUTINE sparsepkl(compute_ka,apy,score,loss,termination,nrec,k)

        USE param, ONLY : zero, one, large  ! Parameters.

        USE initpkl, ONLY : &               ! Initialization of PKL parameters.
            rf, &                           ! Switch for loss function
            p, &                            ! Predicted scores
            y, &                            ! Scores
            indnz, &                        ! array to store indices of k maximum terms of array myx
            init_pklpar, &                  ! S  Furher initialization of parameters.
            def_pklpar                      ! S  Default values of rankinging parameters.

        USE initlmbdca, ONLY : &            ! Initialization of LMB-DCA
            n, &                            ! Number of variables n=nrec
            mcu, &                          ! Maximum number of stored corrections.
            mcinit, &                       ! Initial maximum number of stored corrections.
            epsl, &                         ! Line search parameter.
            xk, &                           ! Fixed point xk for DCA.
            g2, &                           ! Subgradient of f2 at xk.
            defaults, &                     ! S  Default values for parameters.
            init_lmbdcapar                  ! S  Further initialization of parameters.
        USE exe_time, ONLY : getime         ! Execution time.

        IMPLICIT NONE
 

        integer, intent(in) :: nrec,k 
        real(kind=prec), intent(inout) :: apy(nrec)
!f2py   depend(nrec) apy

        integer, intent(inout) :: termination
        character (len=*), intent (inout) :: loss
        real(KIND=prec), dimension(:), intent(in) :: score ! Y
        real(KIND=prec) :: &
            f
        real :: &
            time1, &
            time5, &
            timef
        integer :: &
            turha=0
        integer :: mc      ! Initial maximum number of stored corrections for LMBM
        INTEGER, DIMENSION(4) :: &
            iout           ! Output integer parameters for LMBM.
                           !   iout(1)   Number of used iterations.
                           !   iout(2)   Number of used function evaluations.
                           !   iout(3)   Number of used subgradient evaluations
                           !   iout(4)   Cause of termination:
                           !               1  - The problem has been solved
                           !                    with desired accuracy.
                           !               2  - Changes in function values < tolf in mtesf
                           !                    subsequent iterations.
                           !               3  - Changes in function value <
                           !                    tolf*small*MAX(|f_k|,|f_(k-1)|,1),
                           !                    where small is the smallest positive number
                           !                    such that 1.0 + small > 1.0.
                           !               4  - Number of function calls > mfe.
                           !               5  - Number of iterations > mit.
                           !               6  - Time limit exceeded.
                           !               7  - f < tolb.
                           !               8  - Failure in attaining the demanded accuracy.
                           !               9  - Termination after DCA (critical point).
                           !              -1  - Two consecutive restarts.
                           !              -2  - Number of restarts > maximum number
                           !                    of restarts.
                           !              -3  - Failure in function or subgradient
                           !                    calculations (assigned by the user).
                           !              -4  -
                           !              -5  - Invalid input parameters.
                           !              -6  - Unspecified error.
                       !   iout(5)   Number of f1 evaluations.
                       !   iout(6)   Number of f2 evaluations.
                       !   iout(7)   Number of subgradient evaluations for f1.
                       !   iout(8)   Number of subgradient evaluations for f2.
    
        EXTERNAL compute_ka
    
        if (turha == 1) then ! This will never happen, but the code does not run without this
            call compute_ka(apy,apy,nrec)
        end if

        n  = nrec       ! Numbers of variables in optimization
        print*,'The desired number of non-zero variables ',k,' and the total number of variables in optimization ',n
        
        ! Switch for the loss function defined in Python
        if (loss == "RLS") then
            !print*,'Learning with KronRLS.'
            rf = 1
        else if (loss == "L1") then
            !print*,'Learning with KronLAD.'
            rf = 3
        else if (loss == "hinge-loss") then
            !print*,'Learning with hinge-loss.'
            rf = 2
        else if (loss == "semi-squared-hinge") then
            !print*,'Learning with semi-squared hinge loss.'
            rf = 4
        else if (loss == "svm-hinge") then
            !print*,'Learning with svm-hinge.'
            rf = 5
        else if (loss == "squared-hinge") then
            !print*,'Learning with squared hinge loss.'
            rf = 6

            
        else        
            print*,'Sorry, no loss function "',loss,'" coded.'
            return
        end if

        allocate(p(n),y(n),indnz(n),xk(n),g2(n)) 
        xk = zero
        g2 = zero
        p = zero
        y = score
        indnz = 0

        CALL init_pklpar()
        CALL def_pklpar()

        CALL getime(time1)

        mc = mcinit        ! Initial maximum number of stored corrections
        CALL defaults()         
        CALL init_lmbdcapar()    

        IF (n <= 0) PRINT*,'n<0'
        IF (epsl >= 0.25_prec) PRINT*,'epsl >= 0.25'
        IF (mcu <= 0) PRINT*,'mcu <= 0'

        CALL lmbdcm(compute_ka,apy,n,mc,f,iout(1),iout(2),iout(3),iout(4))
        termination = iout(4) ! return termination criterion to python

        CALL getime(time5)
        timef=time5-time1
        deallocate(p,y,indnz,xk,g2)

        RETURN

    END SUBROUTINE sparsepkl

END MODULE fmodule
