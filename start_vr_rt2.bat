@echo off
rem Start the VR app with full RT2 rendering (real HDRI sky, about 37-55 fps). Same as
rem "start_vr.bat rt2"; extra arguments go to vrsim_cpp.exe.
call "%~dp0start_vr.bat" rt2 %*
