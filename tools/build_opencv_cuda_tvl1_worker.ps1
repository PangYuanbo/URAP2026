param(
    [string]$RepoRoot = "C:\Users\aaron\Desktop\URAP",
    [string]$RunRoot = "C:\Users\aaron\Desktop\URAP\artifacts\opencv_cuda_tvl1_build"
)
$ErrorActionPreference = "Stop"
$sourceRoot = Join-Path $RepoRoot "third_party\opencv-cuda-src"
$opencv = Join-Path $sourceRoot "opencv"
$contrib = Join-Path $sourceRoot "opencv_contrib\modules"
$build = Join-Path $RunRoot "build"
$install = Join-Path $RunRoot "install"
$python = Join-Path $RepoRoot "artifacts\venvs\nps_flow_gpu\Scripts\python.exe"
$status = Join-Path $RunRoot "status.json"
New-Item -ItemType Directory -Force -Path $RunRoot, $build, $install | Out-Null
function Write-Status([string]$phase, [string]$state, [string]$detail) {
    $payload = [ordered]@{ phase=$phase; state=$state; detail=$detail; timestamp=(Get-Date).ToString("o"); build=$build; install=$install }
    $payload | ConvertTo-Json | Set-Content -LiteralPath $status -Encoding utf8
}
try {
    Write-Status "configure" "running" "Configuring OpenCV 4.13 CUDA modules for RTX 5090 sm_120"
    $numpyInclude = & $python -c "import numpy; print(numpy.get_include())"
    $pythonInclude = & $python -c "import sysconfig; print(sysconfig.get_paths()['include'])"
    $pythonLib = & $python -c "import sys, sysconfig, pathlib; print(pathlib.Path(sys.base_prefix)/'libs'/('python'+str(sys.version_info.major)+str(sys.version_info.minor)+'.lib'))"
    $sitePackages = & $python -c "import site; print(site.getsitepackages()[0])"
    $opencvCmake = $opencv.Replace("\", "/")
    $contribCmake = $contrib.Replace("\", "/")
    $buildCmake = $build.Replace("\", "/")
    $installCmake = $install.Replace("\", "/")
    $pythonCmake = $python.Replace("\", "/")
    $numpyInclude = $numpyInclude.Replace("\", "/")
    $pythonInclude = $pythonInclude.Replace("\", "/")
    $pythonLib = $pythonLib.Replace("\", "/")
    $sitePackages = $sitePackages.Replace("\", "/")
    $cmakeArgs = @(
        "-S", $opencvCmake,
        "-B", $buildCmake,
        "-G", "Visual Studio 17 2022",
        "-A", "x64",
        "-DOPENCV_EXTRA_MODULES_PATH=$contribCmake",
        "-DCMAKE_INSTALL_PREFIX=$installCmake",
        "-DBUILD_LIST=core,imgproc,imgcodecs,videoio,calib3d,video,optflow,cudev,cudaarithm,cudaimgproc,cudawarping,cudaoptflow,python3",
        "-DWITH_CUDA=ON",
        "-DCUDA_ARCH_BIN=12.0",
        "-DCUDA_ARCH_PTX=",
        "-DWITH_CUBLAS=ON",
        "-DENABLE_FAST_MATH=ON",
        "-DCUDA_FAST_MATH=ON",
        "-DBUILD_TESTS=OFF",
        "-DBUILD_PERF_TESTS=OFF",
        "-DBUILD_EXAMPLES=OFF",
        "-DBUILD_DOCS=OFF",
        "-DBUILD_JAVA=OFF",
        "-DBUILD_opencv_apps=OFF",
        "-DBUILD_opencv_python3=ON",
        "-DBUILD_opencv_python2=OFF",
        "-DPYTHON3_EXECUTABLE=$pythonCmake",
        "-DPYTHON3_INCLUDE_DIR=$pythonInclude",
        "-DPYTHON3_LIBRARY=$pythonLib",
        "-DPYTHON3_NUMPY_INCLUDE_DIRS=$numpyInclude",
        "-DOPENCV_PYTHON3_INSTALL_PATH=$sitePackages",
        "-DWITH_OPENCL=OFF",
        "-DWITH_IPP=ON"
    )
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed with exit code $LASTEXITCODE" }
    Write-Status "build" "running" "Building Release with MSBuild parallelism"
    & cmake --build $build --config Release --target INSTALL -- /m:16
    if ($LASTEXITCODE -ne 0) { throw "CMake build failed with exit code $LASTEXITCODE" }
    Write-Status "verify" "running" "Verifying CUDA Python bindings"
    $env:PATH = "$install\x64\vc17\bin;$env:PATH"
    & $python -c "import cv2; print(cv2.__version__); print(cv2.cuda.getCudaEnabledDeviceCount()); print(hasattr(cv2.cuda, 'OpticalFlowDual_TVL1_create')); print([x for x in dir(cv2.cuda) if 'TVL1' in x or 'OpticalFlowDual' in x])"
    if ($LASTEXITCODE -ne 0) { throw "CUDA OpenCV verification failed with exit code $LASTEXITCODE" }
    Write-Status "complete" "completed" "CUDA OpenCV build and verification completed"
} catch {
    Write-Status "failed" "failed" $_.Exception.Message
    throw
}