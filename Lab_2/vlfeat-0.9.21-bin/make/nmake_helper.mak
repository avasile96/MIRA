# --------------------------------------------------------------------
#                                         Detect Visual Studio version
# --------------------------------------------------------------------

!IF "$(MSVSVER)" == ""
!IF "$(_NMAKE_VER)" == ""
VL_MSVC = 4.0
VL_MSVS = 40
!ERROR *** Failed to determine version of Visual C++
!ELSEIF "$(_NMAKE_VER)" == "162"
VL_MSVC = 5.0
VL_MSVS = 50
!ERROR *** Detected Visual C++ 5.0 - NOT SUPPORTED
!ELSEIF "$(_NMAKE_VER)" == "6.00.8168.0"
VL_MSVC = 6.0
VL_MSVS = 60
VL_MSC = 1200
!ERROR *** Detected Visual C++ 6.0 - NOT SUPPORTED
!ELSEIF "$(_NMAKE_VER)" == "7.00.9466"
VL_MSVC = 7.0
VL_MSVS = 70
VL_MSC = 1300
!ERROR *** Detected Visual C++ 7.0 - NOT SUPPORTED
!ELSEIF "$(_NMAKE_VER)" == "7.10.3077"
VL_MSVC = 7.1
VL_MSVS = 71
VL_MSC = 1310
!ERROR *** Detected Visual C++ 7.1 - NOT SUPPORTED
!ELSEIF "$(_NMAKE_VER)" == "8.00.50727.42"
VL_MSVC = 8.0
VL_MSVS = 80
VL_MSC = 1400
!ERROR *** Detected Visual C++ 8.0 - NOT SUPPORTED
!ELSEIF "$(_NMAKE_VER)" == "8.00.50727.762"
VL_MSVC = 8.0
VL_MSVS = 80
VL_MSC = 1400
!ERROR *** Detected Visual C++ 8.0 - NOT SUPPORTED
!ELSEIF "$(_NMAKE_VER)" == "9.00.21022.08"
VL_MSVC = 9.0
VL_MSVS = 90
VL_MSC = 1500
!ELSEIF "$(_NMAKE_VER)" == "9.00.30729.01"
VL_MSVC = 9.0
VL_MSVS = 90
VL_MSC = 1500
!ELSEIF "$(_NMAKE_VER)" == "10.00.30128.01"
VL_MSVC = 10.0
VL_MSVS = 100
VL_MSC = 1600
!ELSEIF "$(_NMAKE_VER)" == "10.00.30319.01"
VL_MSVC = 10.0
VL_MSVS = 100
VL_MSC = 1600
!ELSEIF "$(_NMAKE_VER)" == "11.00.40825.2"
VL_MSVC = 11.0
VL_MSVS = 110
VL_MSC = 1700
!ELSEIF "$(_NMAKE_VER)" == "11.00.51106.1"
VL_MSVC = 11.0
VL_MSVS = 110
VL_MSC = 1700
!ELSEIF "$(_NMAKE_VER)" == "11.00.50727.1"
VL_MSVC = 11.0
VL_MSVS = 110
VL_MSC = 1700
!ELSEIF "$(_NMAKE_VER)" == "11.00.60315.1"
VL_MSVC = 11.0
VL_MSVS = 110
VL_MSC = 1700
!ELSEIF "$(_NMAKE_VER)" == "11.00.60430.2"
VL_MSVC = 11.0
VL_MSVS = 110
VL_MSC = 1700
!ELSEIF "$(_NMAKE_VER)" == "11.00.60521.0"
VL_MSVC = 11.0
VL_MSVS = 110
VL_MSC = 1700
!ELSEIF "$(_NMAKE_VER)" == "11.00.60610.1"
VL_MSVC = 11.0
VL_MSVS = 110
VL_MSC = 1700
!ELSEIF "$(_NMAKE_VER)" == "12.00.21005.1"
VL_MSVC = 12.0
VL_MSVS = 120
VL_MSC = 1800
!ELSEIF "$(_NMAKE_VER)" == "14.00.22816.0"
VL_MSVC = 14.0
VL_MSVS = 140
VL_MSC = 1900
!ELSEIF "$(_NMAKE_VER)" == "14.00.23026.0"
VL_MSVC = 14.0
VL_MSVS = 140
VL_MSC = 1900
!ELSEIF "$(_NMAKE_VER)" == "14.00.23506.0"
VL_MSVC = 14.0
VL_MSVS = 140
VL_MSC = 1900
!ELSEIF "$(_NMAKE_VER)" == "14.00.24210.0"
VL_MSVC = 14.0
VL_MSC = 1900
!ELSE
VL_MSVC =
VL_MSVS =
VL_MSC =
!ENDIF
MSVSVER=$(VL_MSVS)
!ENDIF

!IF "$(MSVSVER)" == ""
!MESSAGE *** Cannot determine Visual C++ version
!ERROR *** Aborting make job
!ELSE
!MESSAGE *** Using Microsoft NMAKE version $(_NMAKE_VER)
!MESSAGE *** Using Microsoft Visual C++ version $(MSVSVER)
!MESSAGE ***
!ENDIF

MSVSYEAR =
!IF "$(MSVSVER)" == "90"
MSVSYEAR = 2008
!ELSEIF "$(MSVSVER)" == "100"
MSVSYEAR = 2010
!ELSEIF "$(MSVSVER)" == "110"
MSVSYEAR = 2012
!ELSEIF "$(MSVSVER)" == "120"
MSVSYEAR = 2013
!ELSEIF "$(MSVSVER)" == "140"
MSVSYEAR = 2015
!ENDIF