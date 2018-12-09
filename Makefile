# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target install/strip
install/strip: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing the project stripped..."
	/usr/local/bin/cmake -DCMAKE_INSTALL_DO_STRIP=1 -P cmake_install.cmake
.PHONY : install/strip

# Special rule for the target install/strip
install/strip/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing the project stripped..."
	/usr/local/bin/cmake -DCMAKE_INSTALL_DO_STRIP=1 -P cmake_install.cmake
.PHONY : install/strip/fast

# Special rule for the target install/local
install/local: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing only the local directory..."
	/usr/local/bin/cmake -DCMAKE_INSTALL_LOCAL_ONLY=1 -P cmake_install.cmake
.PHONY : install/local

# Special rule for the target install/local
install/local/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Installing only the local directory..."
	/usr/local/bin/cmake -DCMAKE_INSTALL_LOCAL_ONLY=1 -P cmake_install.cmake
.PHONY : install/local/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/local/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/local/bin/ccmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# Special rule for the target install
install: preinstall
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/usr/local/bin/cmake -P cmake_install.cmake
.PHONY : install

# Special rule for the target install
install/fast: preinstall/fast
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Install the project..."
	/usr/local/bin/cmake -P cmake_install.cmake
.PHONY : install/fast

# Special rule for the target list_install_components
list_install_components:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Available install components are: \"Unspecified\""
.PHONY : list_install_components

# Special rule for the target list_install_components
list_install_components/fast: list_install_components

.PHONY : list_install_components/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/rmolines/github/imgseg/projeto/segmentacao_sequencial/CMakeFiles /home/rmolines/github/imgseg/projeto/segmentacao_sequencial/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/rmolines/github/imgseg/projeto/segmentacao_sequencial/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named gpu_seg

# Build rule for target.
gpu_seg: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 gpu_seg
.PHONY : gpu_seg

# fast build rule for target.
gpu_seg/fast:
	$(MAKE) -f CMakeFiles/gpu_seg.dir/build.make CMakeFiles/gpu_seg.dir/build
.PHONY : gpu_seg/fast

imagem.o: imagem.cpp.o

.PHONY : imagem.o

# target to build an object file
imagem.cpp.o:
	$(MAKE) -f CMakeFiles/gpu_seg.dir/build.make CMakeFiles/gpu_seg.dir/imagem.cpp.o
.PHONY : imagem.cpp.o

imagem.i: imagem.cpp.i

.PHONY : imagem.i

# target to preprocess a source file
imagem.cpp.i:
	$(MAKE) -f CMakeFiles/gpu_seg.dir/build.make CMakeFiles/gpu_seg.dir/imagem.cpp.i
.PHONY : imagem.cpp.i

imagem.s: imagem.cpp.s

.PHONY : imagem.s

# target to generate assembly for a file
imagem.cpp.s:
	$(MAKE) -f CMakeFiles/gpu_seg.dir/build.make CMakeFiles/gpu_seg.dir/imagem.cpp.s
.PHONY : imagem.cpp.s

main.o: main.cu.o

.PHONY : main.o

# target to build an object file
main.cu.o:
	$(MAKE) -f CMakeFiles/gpu_seg.dir/build.make CMakeFiles/gpu_seg.dir/main.cu.o
.PHONY : main.cu.o

main.i: main.cu.i

.PHONY : main.i

# target to preprocess a source file
main.cu.i:
	$(MAKE) -f CMakeFiles/gpu_seg.dir/build.make CMakeFiles/gpu_seg.dir/main.cu.i
.PHONY : main.cu.i

main.s: main.cu.s

.PHONY : main.s

# target to generate assembly for a file
main.cu.s:
	$(MAKE) -f CMakeFiles/gpu_seg.dir/build.make CMakeFiles/gpu_seg.dir/main.cu.s
.PHONY : main.cu.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... install/strip"
	@echo "... install/local"
	@echo "... gpu_seg"
	@echo "... rebuild_cache"
	@echo "... edit_cache"
	@echo "... install"
	@echo "... list_install_components"
	@echo "... imagem.o"
	@echo "... imagem.i"
	@echo "... imagem.s"
	@echo "... main.o"
	@echo "... main.i"
	@echo "... main.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

