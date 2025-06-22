# ROS 

## 📌 Introduction

This is a ROS (Robot Operating System) project. ROS is a flexible framework for writing robot software, providing tools and libraries to help developers create robot applications quickly and efficiently.

This repository provides a basic introduction to ROS and can be used by students for applications such as robot control, simulation experiments, and related research projects.

## 📁 Project Structure

```
WorkSpace
├── build/                   # Compilation space (CMake & catkin cache, configs, temp files)
├── devel/              	 # Development space (compiled files, headers, libraries, executables)
└── src/                	 # Source code
    ├── CMakeLists.txt  	 # Basic build configuration
    └── package/        	 # ROS package (basic ROS unit)
        ├── CMakeLists.txt   # Compilation rules (sources, dependencies, targets)
        ├── package.xml      # Package metadata (name, version, author, dependencies)
        ├── scripts/         # Python scripts
        ├── src/             # C++ source files
        ├── include/         # Header files
        ├── msg/             # Message format files
        ├── srv/             # Service format files
        ├── action/          # Action format files
        ├── launch/          # Launch files (run multiple nodes at once)
        └── config/          # Configuration files
```

## ⚠️ Notes

After building your workspace, run **`source devel/setup.bash`** to update your environment variables so ROS can find your newly built messages and packages.

## 📦 Implementation of HelloWorld Program

### 1. Create and initialize workspace

```
mkdir -p <workspace_name>/src
cd <workspace_name>
catkin_make
```

The above command will first create a workspace and a src subdirectory, and then enter the workspace to call the catkin_make command to compile.

### 2. Enter the src folder to create a ROS package and add dependencies

```
cd src
catkin_create_pkg <ros_package_name> roscpp rospy std_msgs
```

The above command will generate a function package in the workspace. This function package depends on roscpp, rospy and std_msgs. roscpp is a library implemented in C++, rospy is a library implemented in Python, and std_msgs is a standard message library. When creating a ROS function package, it is generally dependent on these three libraries.

### 3. Enter the ROS package, add the scripts directory and edit the python file

```
cd <ros_package_name>
mkdir scripts
```

### 4. Create a new python file

```py
# NAME: HelloWorld.py
import rospy

if __name__ == "__main__":
    rospy.init_node("Hello")
    rospy.loginfo("Hello World!!!!")
```

### 5. Add executable permissions to the python file

```
chmod +x <HelloWorld.py>
or
chmod +x *.py
```

### 6. Edit the CamkeList.txt file under the ROS package

```
catkin_install_python(
	PROGRAMS 
	scripts/<HelloWorld.py>
  	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
```

### 7. Enter the workspace directory and compile

```
cd <workspace_name>
catkin_make
```

### 8. Enter the workspace directory and execute

```
roscore
```

```
cd <workspace_name>
source ./devel/setup.bash
rosrun <ros_package_name> <HelloWorld.py>
```

## 🌟 Custom ROS Message Setup

This guide walks you through creating a custom ROS message (`Person.msg`) and configuring your package to build and use it.

### 1. Define the Message File

Create a directory named `msg` inside your package folder, and add a file named `Person.msg` with the following content:

```plaintext
string name
uint16 age
float64 height
```

### 2.Update package.xml

Add dependencies for message generation and runtime:

```
<build_depend>message_generation</build_depend>
<exec_depend>message_runtime</exec_depend>
```

### 3.Modify CMakeLists.txt

Make sure you find the required packages including `message_generation`:

```
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)
```

Add your message file(s):

```
add_message_files(
  FILES
  Person.msg
)
```

Specify dependencies for generating messages:

```
generate_messages(
  DEPENDENCIES
  std_msgs
)
```

Declare runtime dependencies in the `catkin_package()` call:

```
catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
)
```

