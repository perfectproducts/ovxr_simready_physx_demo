@echo off
rem Start the VR app: the Python sidecar (voice, catalog, downloads) in its own window, then the
rem C++ app (ovxr + ovphysx) once the sidecar listens on port 47800.
rem
rem   start_vr.bat                  Minimal rendering (72 fps, faked sky)
rem   start_vr.bat rt2              full RT2 rendering (real HDRI sky, about 37 fps)
rem   start_vr.bat minimal --no-foveation   extra arguments go to vrsim_cpp.exe
rem
rem Set VRSIM_MIC to choose another microphone (default: the Quest headset mic over Link).
setlocal
cd /d "%~dp0"

set "RENDER=%~1"
if "%RENDER%"=="" set "RENDER=minimal"
set "EXTRA="
shift
:collect
if "%~1"=="" goto collected
set "EXTRA=%EXTRA% %1"
shift
goto collect
:collected

if "%VRSIM_MIC%"=="" set "VRSIM_MIC=Headset Microphone (Oculus"

if not exist "cpp\build\vrsim_cpp.exe" (
    echo cpp\build\vrsim_cpp.exe not found. Build it first: cpp\scripts\build.bat
    goto fail
)
where uv >nul 2>&1 || (
    echo uv not found on PATH. Install it from https://docs.astral.sh/uv/
    goto fail
)

netstat -ano | findstr /r /c:":47800  *0.0.0.0:0 " >nul && (
    echo A sidecar is already listening on port 47800. Close its window first.
    goto fail
)

echo Starting sidecar (mic: %VRSIM_MIC%) ...
start "VRSim sidecar" cmd /k uv run python -m vrsim.sidecar --mic "%VRSIM_MIC%"

rem Wait for the listening socket (remote 0.0.0.0:0; netstat state names are localized). Don't
rem connect to probe it: the sidecar takes the first connection as the app.
set /a TRIES=0
:wait
netstat -ano | findstr /r /c:":47800  *0.0.0.0:0 " >nul && goto ready
set /a TRIES+=1
if %TRIES% geq 180 (
    echo Sidecar did not start within 3 minutes. Check the "VRSim sidecar" window.
    goto fail
)
ping -n 2 127.0.0.1 >nul
goto wait

:ready
echo Starting VR app (render: %RENDER%%EXTRA%) ...
echo Put on the Quest with Link active. The first start under Link can take minutes (shader compile).
cpp\build\vrsim_cpp.exe --render %RENDER%%EXTRA%
echo VR app exited. Close the "VRSim sidecar" window to stop voice and the catalog.
pause
endlocal
goto :eof

:fail
rem Keep a double-clicked window open so the message stays readable.
pause
endlocal
exit /b 1
