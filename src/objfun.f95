!*************************************************************************
!*                                                                       *
!*     sparsePKL --- objective function                                  *
!*                                                                       *
!*     Computation of the value of the DC objective functions J1 and J2  *
!*     and the corresbonding subgradients. Computation of DCA function.  *  
!*     (last modified 29.08.2023 by Napsu).                              *
!*                                                                       *
!*     Possible loss functions:                                          *
!*                                                                       *
!*     - KronRLS: Regularized least squares;                             *
!*     - KronLAD: Absolute value error;                                  *
!*     - hinge-loss: epsilon intensive hinge-loss;                       *
!*     - semi-squared hinge-loss: epsilon intensive hinge-loss           *
!*         with squared difference of predictions and labels;            *
!*     - squared hinge-loss: squared epsilon intensive hinge-loss        *
!*                                                                       *
!*     Possible regularization functions:                                *
!*                                                                       *
!*     - L0 -regularization;                                             *
!*     - double regularization with L1- and L0-norms.                    *
!*                                                                       *
!*     The sparsePKL software is covered by the MIT license.             *                                                *
!*                                                                       *
!*                                                                       *
!*************************************************************************
!*
!*     Modules included:
!*
!*     obj_fun         !
!*

MODULE obj_fun  ! Computation of the value and the subgradient of the 
                ! objective function.

    USE r_precision, ONLY : prec  ! Precision for reals.
    IMPLICIT NONE

    PUBLIC :: &
        myf,  &     ! Computation of the value of the objective.
        myg,  &     ! Computation of the subgradient of the objective.
        myf1, &     ! Computation of the value of the objective f1.
        myg1, &     ! Computation of the subgradient of the objective f1.
        myf2, &     ! Computation of the value of the objective f2.
        dcaf, &     ! Computation of the dca function.
        dcag, &     ! Computation of the subgradient of the dca function.
        sum_of_k_max_elements ! Computation of k-norm.


CONTAINS
    !************************************************************************
    !*                                                                      *
    !*     * SUBROUTINE dcaf *                                              *
    !*                                                                      *
    !*     Computation of the value of f1(x)-(f2(x_k)+g2_k^T (x-x_k)).      *
    !*                                                                      *
    !************************************************************************

    SUBROUTINE dcaf(compute_ka,n,x,f,iterm)

        USE initlmbdca, ONLY : &
            xk, &         ! Saved point xk.
            f2, &         ! Saved value for the second component fuction at xk.
            g2            ! Saved subgradient for f2 at xk.

        IMPLICIT NONE

        ! Array Arguments
        REAL(KIND=prec), DIMENSION(n), INTENT(INOUT) :: &
            x             ! Vector of variables.
        
        ! Scalar Arguments
        REAL(KIND=prec), INTENT(OUT) :: f  ! Value of the dca function.
        INTEGER, INTENT(IN) :: n           ! Number of variables.
        INTEGER, INTENT(OUT) :: iterm      ! Cause of termination:
                                           !   0  - Everything is ok.
                                           !  -3  - Failure in function calculations
        integer :: test_ka = 0 
        EXTERNAL compute_ka

        if (test_ka==1) then ! This will never happen, but the code does not run without this
            call compute_ka(x,x,n)
        end if  
    
        iterm = 0
    
        ! value of the DCA function
        CALL myf1(compute_ka,n,x,f,iterm)
        f = f - f2 - DOT_PRODUCT(g2,x-xk)
        
    END SUBROUTINE dcaf


    !************************************************************************
    !*                                                                      *
    !*     * SUBROUTINE dcag *                                              *
    !*                                                                      *
    !*     Subgradient of f1(x)-(f2(x_k)+g2_k^T (x-x_k)).                   *
    !*                                                                      *
    !************************************************************************

    SUBROUTINE dcag(compute_ka,n,x,grad,iterm)

        USE initlmbdca, ONLY : &
            g2            ! Saved subgradient for f2 at xk.
        USE param, ONLY : zero,one,two,large  ! Parameters.
        IMPLICIT NONE

        ! Array Arguments
        REAL(KIND=prec), DIMENSION(n), INTENT(INOUT) :: &
            x             ! Vector of variables.
        REAL(KIND=prec), DIMENSION(n), INTENT(OUT) :: &
            grad          ! Subgradient.
        
        ! Scalar Arguments
        INTEGER, INTENT(IN) :: n           ! Number of variables.
        INTEGER, INTENT(OUT) :: iterm      ! Cause of termination:
                                           !   0  - Everything is ok.
                                           !  -3  - Failure in function calculations
        integer :: test_ka = 0 
        EXTERNAL compute_ka
    
        if (test_ka==1) then ! This will never happen, but the code does not run without this
            call compute_ka(x,x,n)
        end if  
       
        
        iterm = 0

        ! subgradient of the DCA function
        CALL myg1(compute_ka,n,x,grad,iterm)
        grad = grad - g2
        
    END SUBROUTINE dcag


    !************************************************************************
    !*                                                                      *
    !*     * SUBROUTINE myf *                                               *
    !*                                                                      *
    !*     Computation of the value of the objective f = f1-f2.             *
    !*                                                                      *
    !************************************************************************
     
    SUBROUTINE myf(compute_ka,n,myx,f,iterm)

        USE param, ONLY : zero,one,large  ! Parameters.

        IMPLICIT NONE

        ! Array Arguments
        REAL(KIND=prec), DIMENSION(n), INTENT(IN) :: &
            myx        ! Vector of (dual) variables.

        ! Scalar Arguments
        REAL(KIND=prec), INTENT(OUT) :: f  ! Value of the function.
        INTEGER, INTENT(IN) :: n           ! Number of variables.
        INTEGER, INTENT(OUT) :: iterm      ! Cause of termination:
                                           !   0  - Everything is ok.
                                           !  -3  - Failure in function calculations
        real(KIND=prec) f_tmp
        integer :: test_ka = 0 
        EXTERNAL compute_ka
    
        if (test_ka==1) then ! This will never happen, but the code does not run without this
            call compute_ka(myx,myx,n)
        end if  

        iterm = 0

        ! Computing the function value
        CALL myf1(compute_ka,n,myx,f,iterm)
        CALL myf2(n,myx,f_tmp,iterm)
        f = f - f_tmp

        ! Error checking.
        IF (f > large) iterm = -3  !
        IF (f < -large) iterm = -3 !

        RETURN
      
    END SUBROUTINE myf


    !************************************************************************
    !*                                                                      *
    !*     * SUBROUTINE myg *                                               *
    !*                                                                      *
    !*     Computation of the subgradient of the objective function.        *
    !*                                                                      *
    !************************************************************************
     
    SUBROUTINE myg(compute_ka,n,myx,g,iterm)

        USE param, ONLY : zero,one,large  ! Parameters.

        IMPLICIT NONE

        ! Array Arguments
        REAL(KIND=prec), DIMENSION(n), INTENT(IN) :: myx  ! Vector of variables.
        REAL(KIND=prec), DIMENSION(n), INTENT(OUT) :: g   ! Subgradient.

        ! Scalar Arguments
        INTEGER, INTENT(IN) :: n                        ! Number of variables.
        INTEGER, INTENT(OUT) :: iterm                   ! Cause of termination:
                                                        !   0  - Everything is ok.
                                                        !  -3  - Failure in subgradient calculations
        REAL(KIND=prec), DIMENSION(n) :: g_tmp
        
        integer :: test_ka = 0 
        EXTERNAL compute_ka
    
        if (test_ka==1) then ! This will never happen, but the code does not run without this
            call compute_ka(myx,myx,n)
        end if  

        iterm = 0
        CALL myg1(compute_ka,n,myx,g,iterm)
        CALL myg2(n,myx,g_tmp,iterm)
        g = g - g_tmp

        RETURN

    END SUBROUTINE myg


    !************************************************************************
    !*                                                                      *
    !*     * SUBROUTINE myf1 *                                              *
    !*                                                                      *
    !*     Computation of the value of the objective f1.                    *
    !*                                                                      *
    !************************************************************************

    SUBROUTINE myf1(compute_ka,n,myx,f,iterm)

    USE param, ONLY : zero,one,large  ! Parameters
    USE initpkl, ONLY : &
        epsilon, &     ! epsilon for epsilon intensive hinge-losses
        rf, &          ! switch for loss function
        ireg, &        ! switch for regularization
        rho, &         ! regularization parameter 
        rho2, &        ! double regularization parameter 
        autolambda, &  ! automated regularization parameter
        y, &           ! labels
        p              ! predictions

    IMPLICIT NONE
    ! Array Arguments
    REAL(KIND=prec), DIMENSION(n), INTENT(IN) :: &
        myx            ! Vector of (dual) variables
!f2py depend(n) p(n)
!f2py depend(n) y(n)

    ! Scalar Arguments
    REAL(KIND=prec), INTENT(OUT) :: f  ! Value of the function.
    INTEGER, INTENT(IN) :: n           ! Number of variables.
    INTEGER, INTENT(OUT) :: iterm      ! Cause of termination:
                                       !   0  - Everything is ok.
                                       !  -3  - Failure in function calculations
    REAL(KIND=prec) :: ydiff,ptmp
    INTEGER :: i
    EXTERNAL compute_ka

    iterm = 0
    f = zero
    ydiff = zero
    ptmp = p(1)
            
    SELECT CASE(rf) ! Select the loss function used                   
        CASE(1) ! KronRLS = L2-norm
            call compute_ka(p,myx,n) ! computes p = K myx
            DO i=1,n
                f = f + (p(i)-y(i))**2
            END DO
            f = 0.5_prec*f
        
        CASE(2) ! epsilon intensive hinge loss
            call compute_ka(p,myx,n) ! computes p = K myx
            DO i=1,n
                f = f + MAX (zero,ABS(p(i)-y(i)) - epsilon)
            END DO
            f = f/REAL(n,prec) 

        CASE(3) ! KronLAD = L1 norm
            call compute_ka(p,myx,n) ! computes p = K myx
            DO i=1,n
                f = f + ABS(p(i)-y(i))
            END DO
            f = f/REAL(n,prec) 
        
        CASE(4) ! semi-squared hinge loss
            call compute_ka(p,myx,n) ! computes p = K myx
            DO i=1,n
                f = f + MAX (zero,(p(i)-y(i))**2 - epsilon)
            END DO
            f = 0.5_prec*f
            f = f/REAL(n,prec) 

        CASE(5) ! SVM with hinge loss
            call compute_ka(p,myx,n) ! computes p = K myx
            DO i=1,n
                f = f + MAX (zero,1 - p(i)*y(i))
            END DO

        CASE(6) ! squared hinge loss
            call compute_ka(p,myx,n) ! computes p = K myx
            DO i=1,n
                f = f + MAX (zero,ABS(p(i)-y(i)) - epsilon)**2
            END DO
            f = 0.5_prec*f
            f = f/REAL(n,prec) 
    
        CASE(7) ! squared SVM
            call compute_ka(p,myx,n) ! computes p = K myx
            DO i=1,n
                f = f + (MAX (zero,1 - p(i)*y(i)))**2
            END DO
            f=0.5_prec*f
            f = f/REAL(n,prec) 

        CASE DEFAULT !
            iterm = -3
            RETURN

    END SELECT

    if (autolambda == 1) then ! automaticly selected regularization
        autolambda = 0
        
        ptmp = sum(abs(myx))
        rho = (f / ptmp)/(REAL(n,prec)**2)
        rho2 = rho 
        if (ireg == 0) then
            f = f + f/(REAL(n,prec)**2)
        else ! Double regularization
            f = f + 2.0_prec*f/(REAL(n,prec)**2)
        end if    
        print*,'The first regularization parameter rho = ',rho
    else
        if (ireg == 0) then
            f = f + rho * sum(abs(myx))
        else ! double regularization
            f = f + (rho + rho2) * sum(abs(myx))
        end if
    end if     
    
    ! Error checking.
    IF (f > large) iterm = -3  !
    IF (f < -large) iterm = -3 !

    RETURN
      
    END SUBROUTINE myf1


    !************************************************************************
    !*                                                                      *
    !*     * SUBROUTINE myf2 *                                              *
    !*                                                                      *
    !*     Computation of the value of the objective f2.                    *
    !*                                                                      *
    !************************************************************************
     
    SUBROUTINE myf2(n,myx,f,iterm)

        USE param, ONLY : zero,large    ! Parameters.
        USE initpkl, ONLY : &
        rho, &      ! regularization parameter 
        k           ! number of nonzero elements allowed in myx

        IMPLICIT NONE

        ! Array Arguments
        REAL(KIND=prec), DIMENSION(n), INTENT(IN) :: &
            myx  ! Vector of variables.

        ! Scalar Arguments
        REAL(KIND=prec), INTENT(OUT) :: f  ! Value of the function.
        INTEGER, INTENT(IN) :: n           ! Number of variables.
        INTEGER, INTENT(OUT) :: iterm      ! Cause of termination:
                                           !   0  - Everything is ok.
                                           !  -3  - Failure in function calculations
                
        iterm = 0

        ! Function evaluation
        if (n>k) then
            f = rho * sum_of_k_max_elements(myx, k)
        else
            f = rho * sum(abs(myx))
        end if
    
        ! Error checking.
        IF (f > large) iterm = -3  !
        IF (f < -large) iterm = -3 !

        RETURN
      
    END SUBROUTINE myf2


    !************************************************************************
    !*                                                                      *
    !*     * SUBROUTINE myg1 *                                              *
    !*                                                                      *
    !*     Computation of the subgradient of the function f1.               *
    !*                                                                      *
    !************************************************************************
     
    SUBROUTINE myg1(compute_ka,n,myx,g,iterm)

        USE param, ONLY : zero,one,large  ! Parameters.
        USE initpkl, ONLY : &
            epsilon, &     ! epsilon for epsilon intensive hinge-losses
            rf, &          ! switch for ranking function.
            ireg, &        ! switch for regularization.
            rho, &         ! regularization parameter
            rho2, &        ! double regularization parameter
            y, &           ! labels
            p              ! predictions
    
            IMPLICIT NONE
    
            ! Array Arguments
            REAL(KIND=prec), DIMENSION(n), INTENT(IN) :: myx  ! Vector of variables.
            REAL(KIND=prec), DIMENSION(n), INTENT(OUT) :: g   ! Subgradient.
    
            ! Scalar Arguments
            INTEGER, INTENT(IN) :: n                        ! Number of variables.
            INTEGER, INTENT(OUT) :: iterm                   ! Cause of termination:
                                                            !   0  - Everything is ok.
                                                            !  -3  - Failure in subgradient calculations
            ! Scalar Arguments
            INTEGER :: i
    
            EXTERNAL compute_ka
                
            iterm = 0
            g = zero
    
            ! Gradient evaluation.
            SELECT CASE(rf) ! Select the loss function used      
                CASE(1) ! KronRLS 
                    g = p - y
                    call compute_ka(g,g,n)

                CASE(2) ! Hinge loss
                    DO i=1,n
                        if (ABS(p(i)-y(i)) > epsilon) then
                            if (p(i)-y(i) >= epsilon) then
                                g(i) = 1.0_prec
                            else
                                g(i) = -1.0_prec
                            end if
                        end if    
                    END DO
                    call compute_ka(g,g,n)
                    g = g/REAL(n,prec)
        
                CASE(3) ! KronLAD

                    DO i=1,n
                        IF (p(i)-y(i) > 0) THEN
                            g(i) = 1.0_prec
                        ELSE IF (p(i)-y(i) < 0) THEN
                            g(i) = -1.0_prec    
                        ELSE
                            g(i) = 0.0_prec    
                        END IF
                    END DO
                    call compute_ka(g,g,n)
                    g = g/REAL(n,prec) 

                CASE(4) ! semi-squared Hinge loss
                    DO i=1,n
                        if ((p(i)-y(i))**2 > epsilon) then
                            g(i) = p(i) - y(i)
                        end if    
                    END DO
                    call compute_ka(g,g,n)
                    g = g/REAL(n,prec)

                CASE(5) ! SVM with hinge loss 
                    DO i=1,n
                        if (1 - p(i)*y(i) > 0) then
                            g(i) = - y(i)
                        end if
                    END DO
                    call compute_ka(g,g,n)
                    
                CASE(6) ! squared hinge loss
                    DO i=1,n
                        if (ABS(p(i)-y(i)) > epsilon) then
                            if (p(i)-y(i) > epsilon) then
                                    g(i) = p(i)-y(i)-epsilon
                            else
                                g(i) = p(i)-y(i)+epsilon
                            end if
                                
                        end if                        
                    END DO
                    call compute_ka(g,g,n)
                    g = g/REAL(n,prec)

                CASE(7) ! squared SVM
                    DO i=1,n
                        if (1 - p(i)*y(i) > 0) then
                            g(i) = - y(i)*(1-p(i)*y(i))
                        end if
                    END DO
                    call compute_ka(g,g,n)
                    g = g/REAL(n,prec)
                CASE DEFAULT !
                    iterm = -3
    
            END SELECT

            ! Regularization
            IF (ireg == 0) THEN ! only the L0-norm
                DO i=1,n
                    IF (myx(i) > 0) THEN
                        g(i) = g(i) + rho
                    ELSE IF (myx(i) < 0) THEN
                        g(i) = g(i) - rho   
                    END IF
                END DO
            ELSE ! double regularization
                DO i=1,n
                    IF (myx(i) > 0) THEN
                        g(i) = g(i) + rho2 + rho
                    ELSE IF (myx(i) < 0) THEN
                        g(i) = g(i) - rho2 - rho   
                    END IF
                END DO
            END IF
            RETURN
    
        END SUBROUTINE myg1


    !************************************************************************
    !*                                                                      *
    !*     * SUBROUTINE myg2 *                                              *
    !*                                                                      *
    !*     Computation of the subgradient of the function f2.               *
    !*                                                                      *
    !************************************************************************
     
    SUBROUTINE myg2(n,myx,g,iterm)
        USE param, ONLY : zero,one,two,large    ! Parameters.
        USE initpkl, ONLY : &
            indnz, &        ! array of indices of max elements in myx
            rho, &          ! regularization parameter ! jos tähän laittaisi listan ja laskisi monta kerralla???
            k               ! Number of nonzero elements in myx

        IMPLICIT NONE

        ! Array Arguments
        REAL(KIND=prec), DIMENSION(n), INTENT(IN) :: myx  ! Vector of variables.
        REAL(KIND=prec), DIMENSION(n), INTENT(OUT) :: g   ! Subgradient.

        ! Scalar Arguments
        INTEGER, INTENT(IN) :: n                  ! Number of variables.
        INTEGER, INTENT(OUT) :: iterm             ! Cause of termination:
                                                  !   0  - Everything is ok.
                                                  !  -3  - Failure in subgradient calculations
                                                  !        (assigned by the user).

        integer :: i 
        iterm=0
        g=zero
        
        if (n>k) then
            do i = 1, k
                if (myx(indnz(i)) > 0) then
                    g(indnz(i)) = rho
                else if (myx(indnz(i)) < 0) then
                    g(indnz(i)) = -rho
                end if
            end do
        else ! No sparsity wanted (this is not the most efficient way of doing this). 
            do i = 1, n
                if (myx(i) > 0) then
                    g(i) = rho
                else if (myx(i) < 0) then
                    g(i) = -rho
                end if
            end do
        end if
       
        RETURN

    END SUBROUTINE myg2
    

    !************************************************************************
    !*                                                                      *
    !*     * FUNCTION sum_of_k_max_elements *                               *
    !*                                                                      *
    !*     Computation of the k-norm.                                       *
    !*                                                                      *
    !************************************************************************

    function sum_of_k_max_elements(arr, k) result(sumk)
        USE initpkl, ONLY : &
            indnz        ! array of indices of max elements in myx

        implicit none
        real(KIND=prec), intent(in) :: arr(:)
        integer, intent(in) :: k
        integer :: n, i, j
        real(KIND=prec) :: sumk, tmp
      
        indnz = 0.0_prec
        n = size(arr) 
        if (k > n) then
          stop 'Error: k exceeds the size of the array'
        endif
      
        tmp = bisection(abs(arr),n-k)
        sumk=0.0
        j=1
        do i=1,n
          if (abs(arr(i))>tmp) then
            sumk=sumk+abs(arr(i))
            indnz(j)=i 
            j=j+1
          end if
        end do
        if (j < k+1) then ! for the special case of even values
          do i=1,n
            if (abs(arr(i))==tmp) then
              sumk=sumk+abs(arr(i))
              indnz(j)=i 
              j=j+1
              if (j>k) exit
            end if
          end do
        end if
    end function sum_of_k_max_elements    

    !*************************************************************************************
    recursive function bisection(a,k,strict,v1,v2,k1,k2) result(val)
!    real recursive function bisection(a,k,strict,v1,v2,k1,k2) result(val)
    !*************************************************************************************
    ! Finding the k-th value of a(:) in increasing order by bisection of a [v1;v2] interval,
    ! mostly without sorting or copying the data (see below)
    ! - At each step we have count(a<=v1) < k <= count(a<=v2)
    ! - If the number of elements in the interval falls below a hard-coded threshold, 
    !   we stop the bisection and explicitly sort the remaining elements.
    !   Drawback: a bit more memory occupation and (limited) data duplication
    !   Advantage: if strict is .true., sorting is more efficient for a small number 
    !              of elements
    ! - If the number of elements in the interval falls below 1/10th the input number
    !   of elements, these elements are copied to a new array.
    !   Drawback: a bit more memory occupation and (limited) data duplication
    !   Advantage: much less elements to count, and more cache friendly
    ! - if strict is .true., the returned value is a value of the input array, otherwise
    !   it is an arbitrary value such that count(a<=val) == k, potentially saving 
    !   some final recursion step
    !
    ! The complexity is O(n*log(n)) 
    !
    ! This part of the code is copied (with very small modifications)
    ! from stackoverflow. 
    !
    ! Author: PierU
    ! Licence: Creative Commons Share Alike license: 
    !   https://creativecommons.org/licenses/by-sa/4.0/
    ! Available at: 
    !   https://stackoverflow.com/questions/75975549/find-and-modify-highest-n-values-in-a-2d-array-in-fortran
    !
    !*************************************************************************************
    real(KIND=prec), intent(in) :: a(:)
    integer, intent(in) :: k
    logical, intent(in),  optional :: strict
    real(KIND=prec), intent(in), optional :: v1, v2
    integer, intent(in), optional :: k1, k2
    real(KIND=prec) :: val

    integer, parameter :: NTOSORT = 10000
    integer :: n, k0, kk1, kk2, i, j
    real(KIND=prec) :: v0, vv1, vv2
!    real(KIND=prec) :: v0, vv1, vv2, c
    logical :: strict___
    real(KIND=prec), allocatable :: b(:)
    !*************************************************************************************
    n = size(a)
    strict___ = .true.; if (present(strict))  strict___ = strict
        
    if (strict___ .and. n <= NTOSORT) then
        b = a(:)
        call quickselect(b,1,n,k)
        val = b(k)
        return
    end if

    if (.not.present(v1)) then
        ! Search for the min value vv1 and max value vv2 (faster than using minval/maxval)
        ! Generally in the code, k = count(a <= v)
        vv1 = a(1); kk1 = 1
        vv2 = a(1); kk2 = n
        do i = 2, n
            if (a(i) >  vv2) then
                vv2 = a(i)
            else if (a(i) <  vv1) then
                vv1 = a(i)
                kk1 = 1
            else if (a(i) == vv1) then
                kk1 = kk1 + 1
            end if
        end do
    
        ! trivial cases
        if (k <= kk1) then
            val = vv1
            return
        end if
        if (k == n) then
            val = vv2
            return
        end if
    else
        vv1 = v1; kk1 = k1
        vv2 = v2; kk2 = k2
    end if
        
    ! Reduce the [v1,v2] interval by bisection
    v0 = 0.5*(vv1+vv2)
    ! if the middle value falls back on v1 or v2, then no middle value can be obtained
    ! we are at the solution
    if (v0 == vv2 .or. v0 == vv1) then 
        val = vv2
    else
        ! actual bisection
        k0 = count(a <= v0)
        if (.not.strict___ .and. k0 == k) then
            ! v0 is not necessarily a value present in a(:)
            val = v0
        else
            if (k0 >= k) then
                vv2 = v0; kk2 = k0
            else
                vv1 = v0; kk1 = k0
            end if
            if (kk2-kk1 <= n/10) then
            ! Copied pure subroutine extract(a,v1,v2,b,m) directly here by Napsu
                allocate( b(kk2-kk1) )
                j = 0
                do i = 1, size(a)
                    if (a(i) > vv1 .and. a(i) <= vv2) then
                        j = j + 1
                        b(j) = a(i)
                    end if
                end do

                val = bisection(b,k-kk1,strict)
                deallocate(b)
            else
                val = bisection(a,k,strict,vv1,vv2,kk1,kk2)
            end if
        end if
    end if
    
    end function bisection
            
    !*************************************************************************************
    recursive subroutine quickselect(a,ifirst,ilast,k)
    !*************************************************************************************
    ! Author: t-nissie
    ! License: GPLv3
    ! Gist: https://gist.github.com/t-nissie/479f0f16966925fa29ea 
    !*************************************************************************************
    real(KIND=prec), intent(inout) :: a(:)
    integer, intent(in) :: ifirst, ilast, k
    real(KIND=prec) :: x, t
    integer n, i, j
    !*************************************************************************************  
    n = size(a)
    x = a( (ifirst+ilast) / 2 )
    i = ifirst
    j = ilast
    do
        do while (a(i) < x)
            i=i+1
        end do
        do while (x < a(j))
            j=j-1
        end do
        if (i >= j) exit
        t = a(i);  a(i) = a(j);  a(j) = t
        i=i+1
        j=j-1
    end do
    if (ifirst < i-1 .and. k <= i-1) call quickselect(a,ifirst,i-1,k)
    if (j+1 < ilast  .and. k >= j+1) call quickselect(a,j+1,ilast,k)
    end subroutine quickselect

END MODULE obj_fun
