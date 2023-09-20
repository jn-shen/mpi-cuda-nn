OBJDIR=obj
INCDIR=inc

# Compilers
CUD=nvcc

TARGET := main

CPPSRCS := $(wildcard *.cpp)
CPPOBJS := $(CPPSRCS:%.cpp=$(OBJDIR)/%.o)

CUDSRCS := $(wildcard *.cu)
CUDOBJS := $(CUDSRCS:%.cu=$(OBJDIR)/%.o)

UTLSRCS := $(wildcard utils/*.cpp) 
UTLOBJS := $(UTLSRCS:utils/%.cpp=$(OBJDIR)/%.o)

OBJS := $(CPPOBJS) $(CUDOBJS) $(UTLOBJS)

# Flags
CUDFLAGS=-O3
INCFLAGS=-I$(INCDIR)
LDFLAGS=-larmadillo -lcublas -lcudart

CU_CMD=$(CUD) $(CUDFLAGS) $(INCFLAGS)

# Using nvcc to compile all object files
$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(OBJDIR)
	$(CU_CMD) -c $< -o $@

$(OBJDIR)/%.o: utils/%.cpp
	@mkdir -p $(OBJDIR)
	$(CU_CMD) -c $< -o $@

$(OBJDIR)/%.o: %.cu
	@mkdir -p $(OBJDIR)
	$(CU_CMD) -c $< -o $@

$(TARGET): $(OBJS)
	$(CUD) $(OBJS) -o $@ $(LDFLAGS)

clean:
	rm -rf $(OBJDIR)/*.o main