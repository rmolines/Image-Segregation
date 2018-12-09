# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rmolines/github/imgseg/projeto/segmentacao_sequencial

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rmolines/github/imgseg/projeto/segmentacao_sequencial

# Include any dependencies generated for this target.
include CMakeFiles/gpu_seg.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/gpu_seg.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/gpu_seg.dir/flags.make

CMakeFiles/gpu_seg.dir/main.cu.o: CMakeFiles/gpu_seg.dir/flags.make
CMakeFiles/gpu_seg.dir/main.cu.o: main.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rmolines/github/imgseg/projeto/segmentacao_sequencial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/gpu_seg.dir/main.cu.o"
	/usr/local/cuda-10.0/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/rmolines/github/imgseg/projeto/segmentacao_sequencial/main.cu -o CMakeFiles/gpu_seg.dir/main.cu.o

CMakeFiles/gpu_seg.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/gpu_seg.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/gpu_seg.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/gpu_seg.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/gpu_seg.dir/imagem.cpp.o: CMakeFiles/gpu_seg.dir/flags.make
CMakeFiles/gpu_seg.dir/imagem.cpp.o: imagem.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rmolines/github/imgseg/projeto/segmentacao_sequencial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/gpu_seg.dir/imagem.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gpu_seg.dir/imagem.cpp.o -c /home/rmolines/github/imgseg/projeto/segmentacao_sequencial/imagem.cpp

CMakeFiles/gpu_seg.dir/imagem.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gpu_seg.dir/imagem.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rmolines/github/imgseg/projeto/segmentacao_sequencial/imagem.cpp > CMakeFiles/gpu_seg.dir/imagem.cpp.i

CMakeFiles/gpu_seg.dir/imagem.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gpu_seg.dir/imagem.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rmolines/github/imgseg/projeto/segmentacao_sequencial/imagem.cpp -o CMakeFiles/gpu_seg.dir/imagem.cpp.s

# Object files for target gpu_seg
gpu_seg_OBJECTS = \
"CMakeFiles/gpu_seg.dir/main.cu.o" \
"CMakeFiles/gpu_seg.dir/imagem.cpp.o"

# External object files for target gpu_seg
gpu_seg_EXTERNAL_OBJECTS =

CMakeFiles/gpu_seg.dir/cmake_device_link.o: CMakeFiles/gpu_seg.dir/main.cu.o
CMakeFiles/gpu_seg.dir/cmake_device_link.o: CMakeFiles/gpu_seg.dir/imagem.cpp.o
CMakeFiles/gpu_seg.dir/cmake_device_link.o: CMakeFiles/gpu_seg.dir/build.make
CMakeFiles/gpu_seg.dir/cmake_device_link.o: /usr/local/cuda-10.0/lib64/libcudart_static.a
CMakeFiles/gpu_seg.dir/cmake_device_link.o: /usr/lib/x86_64-linux-gnu/librt.so
CMakeFiles/gpu_seg.dir/cmake_device_link.o: CMakeFiles/gpu_seg.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rmolines/github/imgseg/projeto/segmentacao_sequencial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA device code CMakeFiles/gpu_seg.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gpu_seg.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gpu_seg.dir/build: CMakeFiles/gpu_seg.dir/cmake_device_link.o

.PHONY : CMakeFiles/gpu_seg.dir/build

# Object files for target gpu_seg
gpu_seg_OBJECTS = \
"CMakeFiles/gpu_seg.dir/main.cu.o" \
"CMakeFiles/gpu_seg.dir/imagem.cpp.o"

# External object files for target gpu_seg
gpu_seg_EXTERNAL_OBJECTS =

gpu_seg: CMakeFiles/gpu_seg.dir/main.cu.o
gpu_seg: CMakeFiles/gpu_seg.dir/imagem.cpp.o
gpu_seg: CMakeFiles/gpu_seg.dir/build.make
gpu_seg: /usr/local/cuda-10.0/lib64/libcudart_static.a
gpu_seg: /usr/lib/x86_64-linux-gnu/librt.so
gpu_seg: CMakeFiles/gpu_seg.dir/cmake_device_link.o
gpu_seg: CMakeFiles/gpu_seg.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rmolines/github/imgseg/projeto/segmentacao_sequencial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable gpu_seg"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gpu_seg.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gpu_seg.dir/build: gpu_seg

.PHONY : CMakeFiles/gpu_seg.dir/build

CMakeFiles/gpu_seg.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gpu_seg.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gpu_seg.dir/clean

CMakeFiles/gpu_seg.dir/depend:
	cd /home/rmolines/github/imgseg/projeto/segmentacao_sequencial && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rmolines/github/imgseg/projeto/segmentacao_sequencial /home/rmolines/github/imgseg/projeto/segmentacao_sequencial /home/rmolines/github/imgseg/projeto/segmentacao_sequencial /home/rmolines/github/imgseg/projeto/segmentacao_sequencial /home/rmolines/github/imgseg/projeto/segmentacao_sequencial/CMakeFiles/gpu_seg.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/gpu_seg.dir/depend
