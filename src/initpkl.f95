!*************************************************************************
!*                                                                       *
!*     Initialization of parameters for sparsePKL                        *
!*     (version 0.1, last modified 11.09.2023)                           *
!*                                                                       *
!*     The sparsePKL software is covered by the MIT license.             *
!*                                                                       *
!*************************************************************************
!*
!*     Modules included:
!*
!*     initpkl          ! Initialization of parameters for pairwise learning.
!*     initlmbdca       ! Initialization of LMB-DCA -solver.
!*

MODULE initpkl  ! Initialization of parameters for pairwise kernel learning.

    USE r_precision, ONLY : prec  ! Precision for reals.
    IMPLICIT NONE

    ! Input real parameters. Give values here.
    REAL(KIND=prec), SAVE :: & !
        epsilon = 0.01_prec           ! Epsilon for epsilon intensive hinge-losses

    ! Allocatable tables
    REAL(KIND=prec), SAVE, DIMENSION(:), allocatable :: &
        y, &                          ! Labels, y(nrecords).
        p                             ! Predicted scores, p(nrecords).
    INTEGER, SAVE, DIMENSION(:), allocatable :: &
        indnz
    
    ! Other Real parameters.
    REAL(KIND=prec), SAVE :: & !
        rho, &
        rho2

    ! Other integer parameters.
    INTEGER, SAVE :: & !
        k, &                         ! Number of nonzero elements allowed 
        rf, &                        ! Switch for loss function (from python).
                                     ! 1 - Pairwise hinge loss with linear functions (RankSVM)
                                     ! default - error
        ireg, &                      ! Switch for regularization (from python).
                                     ! 0 - L0-norm   
                                     ! 1 - L1 + L0 norm 
        autolambda, &                ! Switch for automated selection of lambda.
        nrecord, &                   ! this value now comes from python
        nft                          ! this value now comes from python


CONTAINS


    SUBROUTINE init_pklpar()    ! User supplied subroutine for further initialization of parameters (when needed).
                                ! May be left empty.
        IMPLICIT NONE

    END SUBROUTINE init_pklpar

    SUBROUTINE def_pklpar()     ! Default values for parameters.

        IMPLICIT NONE
        !IF (nrecord <= 1000) THEN

        !ELSE IF (nrecord <= 10000) THEN

        !ELSE

        !END IF

    END SUBROUTINE def_pklpar

END MODULE initpkl


MODULE initlmbdca  ! Initialization of parameters for LMB-DCA.

    USE r_precision, ONLY : prec   ! Precision for reals.
    USE param, ONLY : zero, one    ! Parameters.

    IMPLICIT NONE

    ! Parameters
    INTEGER, PARAMETER :: &
        na      = 2, &             ! Maximum bundle dimension, na >= 2.
        mcu     = 15, &            ! Maximum number of stored corrections, mcu >=1.
        mcinit  = 7, &             ! Initial maximum number of stored corrections, mcu >= mcinit >= 3.
                                   ! If mcinit <= 0, the default value mcinit = 3 will be used.
                                   ! However, the value mcinit = 7 is recommented.
        maxit   = 100, &           ! Max number of iterations in dca, maxit > 0. 
                                   !   maxit >= 1 if imet = 0,    
                                   !   maxit > 100, otherwise.
        inma    = 1, &             ! Selection of line search method:
                                   !   inma = 0, weak Wolfe line search.
                                   !   inma = 1, nonmonotone  weak Wolfe line search.
        mnma    = 10, &            ! Maximum number of function values used in nonmonotone line search.
        maxnin  = 20               ! Maximum number of interpolations, maxnin >= 0.
                                   ! For example:
                                   !   inma = 0, maxnin = 200.
                                   !   inma = 1, mnma=10, maxnin = 20.


    ! Real parameters (if parameter value <= 0.0 the default value of the parameter will be used).
    REAL(KIND=prec), SAVE :: &
        tolb    = zero, &         ! Tolerance for the function value (default = -large).
        tolf    = 1.0E-08_prec, & ! Tolerance for change of function values (default = 1.0E-8).
        tolf2   = -10.0_prec, &   ! Second tolerance for change of function values.
                                  !   - If tolf2 < 0 the the parameter and the corresponding termination
                                  !   criterion will be ignored (recommended with inma=1).
                                  !   - If tolf2 = 0 the default value 1.0E+4 will be used.
        tolg    = 1.0E-6_prec, &  ! Tolerance for the termination criterion (default = 1.0E-5).
        tolg2   = 1.0E-5_prec, &  ! Tolerance for the second termination criterion (default = 1.0E-3).
        eta     = 0.5_prec, &     ! Distance measure parameter, eta > 0.
                                  !   - If eta < 0  the default value 0.0001 will be used.
        epsl    = 0.24E+00, &     ! Line search parameter, 0 < epsl < 0.25 (default = 0.24).
        xmax    = 1000.0_prec     ! Maximum stepsize, 1 < XMAX (default = 1000).

    ! Integer parameters (if value <= 0 the default value of the parameter will be used).
    INTEGER, SAVE :: &
        imet, &                   ! Selection of method (in python):
                                  !   imet = 0, lmbdca (default),
                                  !   imet = 1, traditional dca with lmbm as an underlying solver.    
        n, &                      ! Number of variables.
        !mit     = 25000, &         ! Maximun number of iterations.
        !mfe     = 500000, &        ! Maximun number of function evaluations
        mit     = 500, &          ! Maximun number of iterations.
        mfe     = 50000, &        ! Maximun number of function evaluations
        mitdca  = 0, &            ! Maximun number of iterations for dca.
                                  ! E.g. mitdca = 100, if imet = 0 and mitdca = 500 if imet = 1.
        mfedca  = 0, &            ! Maximun number of function evaluations fordca.
                                  ! E.g. mfedca = 500, if imet = 0 and mfedca = 5000 if imet = 1.
        mtesf   = 0, &            ! Maximum number of iterations with changes of
                                  ! function values smaller than tolf (default = 10).
        iprint  = 1, &            ! Printout specification:
                                  !    -1  - No printout.
                                  !     0  - Only the error messages.
                                  !     1  - The final values of the objective function
                                  !          (default used if iprint < -1).
                                  !     2  - The final values of the objective function and the
                                  !          most serious warning messages.
                                  !     3  - The whole final solution.
                                  !     4  - At each iteration values of the objective function.
                                  !     5  - At each iteration the whole solution
        ipdca   = 1, &            ! Printout specification for DCA (see values above).
        iscale  = 0               ! Selection of the scaling with LMBM:
                                  !     0  - Scaling at every iteration with STU/UTU (default).
                                  !     1  - Scaling at every iteration with STS/STU.
                                  !     2  - Interval scaling with STU/UTU.
                                  !     3  - Interval scaling with STS/STU.
                                  !     4  - Preliminary scaling with STU/UTU.
                                  !     5  - Preliminary scaling with STS/STU.
                                  !     6  - No scaling.

    ! Allocatable tables
    REAL(KIND=prec), SAVE, DIMENSION(:), allocatable :: &
        myx, &                    ! Vector of variables, myx(nrecords)
        xk, &                     ! Fixed point xk for DCA.
        g2                        ! Subgradient of f2 at xk.
    REAL(KIND=prec), SAVE :: &
        f2                        ! Value of f2 at xk. 


CONTAINS

    SUBROUTINE defaults()  ! Default values for parameters.

        USE param, ONLY: small, large, zero, one, half
        IMPLICIT NONE

        IF (iprint < -1) iprint   = 1               ! Printout specification.
        IF (ipdca < -1) ipdca     = 1               ! Printout specification for DCA.
        IF (mit   <= 0) mit       = 500             ! Maximum number of iterations.
        IF (mfe   <= 0) mfe       = 100*mit         ! Maximum number of function evaluations.
        IF (mitdca <= 0) THEN                       ! Maximum number of iterations in DCA.
            IF (imet == 0) THEN ! LMB-DCA
                mitdca   = 100                      
            ELSE ! DCA    
                mitdca   = 500                      
            END IF
        END IF
        IF (mfedca <= 0) THEN                       ! Maximum number of function evaluations in DCA.
            IF (imet == 0) THEN ! LMB-DCA
                mfedca   = 500                      
            ELSE ! DCA    
                mfedca   = 5000                      
            END IF
        END IF
        IF (tolf  <= zero) tolf   = 1.0E-08_prec    ! Tolerance for change of function values.
        IF (tolf2 == zero) tolf2  = 1.0E+04_prec    ! Second tolerance for change of function values.
        IF (tolb  == zero) tolb   = -large + small  ! Tolerance for the function value.
        IF (tolg  <= zero) tolg   = 1.0E-05_prec    ! Tolerance for the termination criterion.
        IF (tolg2  <= zero) tolg2  = 1.0E-03_prec   ! Tolerance for the second termination criterion.
        IF (xmax  <= zero) xmax   = 1.5_prec        ! Maximum stepsize.
        IF (eta   <  zero) eta    = half            ! Distance measure parameter
        IF (epsl  <= zero) epsl   = 1.0E-04_prec    ! Line search parameter,
        IF (mtesf <= 0) mtesf     = 10              ! Maximum number of iterations with changes
                                                    ! of function values smaller than tolf.
        IF (iscale > 6 .OR. iscale < 0) iscale = 0  ! Selection of the scaling.


    END SUBROUTINE defaults

    SUBROUTINE init_lmbdcapar()  ! User supplied subroutine for further initialization of parameters
                                 ! (when needed) for LMB-DCA. May be left empty.
        IMPLICIT NONE

    END SUBROUTINE init_lmbdcapar

END MODULE initlmbdca
