compiler= $(shell mpicc --showme:command)

CC=mpicc

CFLAGS=
ifeq ($(compiler),icc)
	CFLAGS+=-ipo -xHost
endif
CFLAGS+=-O3

ifdef develop
	ifeq ($(develop),yes)
		CFLAGS+=-g 
		CFLAGS+=-Wall -Wextra		
	endif
endif
ifdef use_double
	ifeq ($(use_double),yes)
		CFLAGS+=-D USE_DOUBLE_PRECISION=double
	endif
endif
CFLAGS+=-fno-omit-frame-pointer

HOST_SIZE   := $(shell getconf LONG_BIT)

#CFLAGS+=-m${HOST_SIZE}
#CFLAGS+=-qopt-report=5
#CFLAGS+=-Wall -Wextra
#CFLAGS+=-Wconversion
#CFLAGS+=-Wno-unused-but-set-variable -Wno-unused-parameter

SRCDIR= src
DEPENCOMMON=$(SRCDIR)/calculosCompartidos.o $(SRCDIR)/fgauss.o $(SRCDIR)/fvoigt.o  $(SRCDIR)/me_der.o $(SRCDIR)/mil_sinrf.o $(SRCDIR)/lib.o $(SRCDIR)/create_cuantic.o $(SRCDIR)/utilsFits.o $(SRCDIR)/milosUtils.o $(SRCDIR)/convolution.o $(SRCDIR)/readConfig.o
DEPEN_SEQ=$(SRCDIR)/milos.o 
LDLIBS= -lm -lcfitsio -lnsl -lgsl -lgslcblas -lfftw3 -ldl -lpthread -L/usr/local/cuda/lib64 -lcudart -lcusolver
BIN= milos  


all: $(BIN) cuda_eigenvalues.o

cuda_eigenvalues.o: 
	$(CC) -c -I/usr/local/cuda/include cuda_eigenvalues.c

milos: $(DEPENCOMMON) $(DEPEN_SEQ) cuda_eigenvalues.o
	$(CC) -o $@ $^ $(CFLAGS) $(LDLIBS) 

clean:
	rm -f  $(SRCDIR)/*.o $(BIN)
