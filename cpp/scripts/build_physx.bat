@echo off
rem Builds the PhysX 5 SDK (CPU-only, release, dynamic CRT) from cpp\_deps\PhysX into
rem cpp\_deps\PhysX\physx\install\vc17win64-cpu-only\PhysX, where cpp\CMakeLists.txt finds it.
rem
rem generate_projects.bat bootstraps NVIDIA "packman", which downloads its own Python and the
rem SDK's pinned build dependencies (CMake, etc.) from NVIDIA's servers before generating the
rem Visual Studio projects.
rem
rem Usage (from the repo root):   cpp\scripts\build_physx.bat

setlocal
set "ROOT=%~dp0..\_deps\PhysX\physx"
if not exist "%ROOT%\generate_projects.bat" (
  echo PhysX sources not found at %ROOT%
  echo Clone them first:  git clone --depth 1 https://github.com/NVIDIA-Omniverse/PhysX.git cpp\_deps\PhysX
  exit /b 1
)

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

pushd "%ROOT%"
call generate_projects.bat vc17win64-cpu-only || (popd & exit /b 1)
msbuild compiler\vc17win64-cpu-only\INSTALL.vcxproj /p:Configuration=release /p:Platform=x64 /m /v:minimal /nologo || (popd & exit /b 1)
popd
echo.
echo PhysX installed to %ROOT%\install\vc17win64-cpu-only\PhysX
