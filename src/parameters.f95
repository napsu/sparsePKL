!*************************************************************************
!*                                                                       *
!*     Precision and parameters for LMB-DCA (last modified 23.08.2023).  *
!*                                                                       *
!*************************************************************************
!*
!*     Modules included:
!*
!*     r_precision      ! Precision for reals.
!*     param            ! Parameters
!*     exe_time         ! Execution time.
!*

MODULE r_precision  ! Precision for reals.

  IMPLICIT NONE
  !INTEGER, PARAMETER, PUBLIC :: prec = SELECTED_REAL_KIND(2*PRECISION(1.0))  ! old DOUBLE PRECISION
  INTEGER, PARAMETER, PUBLIC :: prec = SELECTED_REAL_KIND(12)
  !INTEGER, PARAMETER, PUBLIC :: prec = 8

END MODULE r_precision


MODULE param  ! Parameters

  USE r_precision, ONLY : prec      ! Precision for reals.
  IMPLICIT NONE

! Intrinsic Functions
!  INTRINSIC TINY,HUGE

! Parameters
  INTEGER, PARAMETER, PUBLIC :: maxeps = 20, maxnrs = 2000
  REAL(KIND=prec), PARAMETER, PUBLIC :: &
       zero    = 0.0_prec,    & ! 
       half    = 0.5_prec,    & ! 
       one     = 1.0_prec,    & ! 
       two     = 2.0_prec,    & !
       large   = 3.40282347*10.**38,  & !
       small   = 1.17549435*10.**(-38)     ! 
!       large   = HUGE(zero),  & !
!       small   = TINY(zero)     ! 

END MODULE param


MODULE exe_time  ! Execution time.
  IMPLICIT NONE

  PUBLIC :: getime

CONTAINS
  SUBROUTINE getime(tim)  ! Execution time.
  IMPLICIT NONE
      
! Scalar Arguments
    REAL, INTENT(OUT):: tim  ! Current time, REAL argument.

! Intrinsic Functions
    INTRINSIC CPU_TIME

    CALL CPU_TIME(tim)

  END SUBROUTINE getime

END MODULE exe_time
