# hmm-cpp

[![Build Status](https://travis-ci.com/day253/hmm-cpp.svg?branch=master)](https://travis-ci.com/day253/hmm-cpp)

## 依赖

- cmake
- googletest
- visual studio code

### ubuntu

```shell
sudo apt-get install cmake g++ gdb
```

### Googletest安装

```shell
git clone --depth=1 https://github.com/google/googletest.git
cd googletest
mkdir build
cd build
cmake ..
make
sudo make install
```

### update cmake

```shell
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
sudo apt-get update
sudo apt-get upgrade
```

## 使用步骤

### 1.cmake

使用`ext install ms-vscode.cpptools`安装Microsoft C/C++ 插件

使用`ext install twxs.cmake`安装CMake 语言插件

使用`ext install ms-vscode.cmake-tools`安装CMake 工具插件

使用`>cmake: Configure`生成编译命令

### 2.本地调试

执行`>Cmake: Run Tests`测试代码。

### 3.Segmentation fault (core dumped)

```
ulimit -c unlimited

sudo sysctl -w kernel.core_pattern=core.%e.%p
```
