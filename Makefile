OBJDIR=obj
INCDIR=inc

# Paths for Armadillo
ARMA_INCLUDE=./lib/armadillo/usr/include
ARMA_LIB=./lib/armadillo/usr/lib64

# Path for Lapack
LAPACK_LIB=./lib/lapack/lib64

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
CUDFLAGS=-O3 -arch=sm_70
INCFLAGS=-I$(INCDIR) -I$(ARMA_INCLUDE) -I${MPI_HOME}/include
LDFLAGS=-L$(ARMA_LIB) -L${MPI_HOME}/lib -L${LAPACK_LIB} -lblas -llapack -lmpi -larmadillo -lcublas -lcudart

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