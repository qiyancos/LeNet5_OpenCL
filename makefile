#=================Format=================#
#target ..:prerequisites ...（目标|依赖）#
#	recipe（命令）                   #
#	...				 #
#=================Format=================#


#source file
SRCS += \
src/sample.cpp \
src/backward.cpp \
src/bmp.cpp \
src/cnn.cpp \
src/forward.cpp \
src/init.cpp \
src/math_functions.cpp \
src/mnist.cpp \
src/model.cpp \
src/predict.cpp \
src/train.cpp \
#src/sample.cpp 
#src/main.cpp 

#prerequisites
OBJS = $(SRCS:.cpp=.o)
DEPS = $(SRCS:.cpp=.d)

#-include subdir.mk

#.开头的target会被排除,没有在依赖链上
#-std=c++              -std = c90|c99|c11
#-Wall		       生成所有警告信息
#-c		       只编译不链接
#fmessage-length=0     换行
#-MMD -MP -MF -MT      生成依赖文件.d
#-o		       指定输出的文件名称=target.o
#$@ $^ $<              目标文件|所有依赖文件|第一个依赖文件                
src/%.o: src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -std=c++11 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

#排除.开头的第一个target为默认最终目标
all: OpenCL_CPU
OpenCL_CPU: $(OBJS)
	@echo 'Building Target: $@'
	@echo $(OBJS)
	g++ -o OpenCL_CPU $(OBJS) -lOpenCL #-lOpenCL用于链接
	@echo 'Finish Building Target: $@'
	@echo ' '

#.PHONY定义目标clean
#和OpenCL_CPU没有依赖关系 直接执行make是不会运行到的
#通过make clean执行
.PHONY: clean
clean:
	-@rm OpenCL_CPU $(OBJS) $(DEPS)
	@echo 'Finished clean'
