@echo off
rem Configure and build cpp\build\vrsim_cpp.exe (Release, Ninja, VS 2022 Build Tools).
rem Prerequisite: the ovphysx C SDK extracted to cpp\_deps\ovphysx-sdk (see cpp\README.md).
setlocal
set "VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
set "PATH=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
set "SRC=%~dp0.."
cmake -S "%SRC%" -B "%SRC%\build" -G Ninja -DCMAKE_BUILD_TYPE=Release || exit /b 1
cmake --build "%SRC%\build" || exit /b 1
echo Built %SRC%\build\vrsim_cpp.exe
