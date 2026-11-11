@echo off
rem Start the Newton edition: the sidecar (voice, catalog, downloads, toaster FMUs) in its own window,
rem then the VR app (ovxr + ovrtx + ovstage + ovnewton) once the sidecar listens on port 47801.
rem
rem   start_newton_vr.bat                    Minimal rendering (faked sky)
rem   start_newton_vr.bat rt2                full RT2 rendering (real HDRI sky)
rem   start_newton_vr.bat minimal --no-foveation   extra arguments go to python -m vrnx
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

where uv >nul 2>&1 || (
    echo uv not found on PATH. Install it from https://docs.astral.sh/uv/
    goto fail
)

netstat -ano | findstr /r /c:":47801  *0.0.0.0:0 " >nul && (
    echo A sidecar is already listening on port 47801. Close its window first.
    goto fail
)

echo Starting sidecar (mic: %VRSIM_MIC%) ...
start "VRSim Newton sidecar" cmd /k uv run python -m vrnx.sidecar --mic "%VRSIM_MIC%"

rem Wait for the listening socket without connecting: the sidecar takes the first connection as the app.
set /a TRIES=0
:wait
netstat -ano | findstr /r /c:":47801  *0.0.0.0:0 " >nul && goto ready
set /a TRIES+=1
if %TRIES% geq 180 (
    echo Sidecar did not start within 3 minutes. Check the "VRSim Newton sidecar" window.
    goto fail
)
ping -n 2 127.0.0.1 >nul
goto wait

:ready
echo Starting VR app (render: %RENDER%%EXTRA%) ...
echo Put on the Quest with Link active. The first start compiles RTX shaders and Warp kernels (minutes).
uv run python -m vrnx --render %RENDER%%EXTRA%
echo VR app exited. Close the "VRSim Newton sidecar" window to stop voice and the catalog.
pause
endlocal
goto :eof

:fail
pause
endlocal
exit /b 1
