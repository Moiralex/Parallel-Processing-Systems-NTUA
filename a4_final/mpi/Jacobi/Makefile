CC=mpicc
CFLAGS=-O3 -lm -Wall
RES=-DPRINT_RESULTS
CONV=-DTEST_CONV

all: jacobi

jacobi: mpi_jacobi.c utils.c
	# $(CC) $(CFLAGS) $(RES) $(CONV) mpi_jacobi.c utils.c -o jacobi
	# $(CC) $(CFLAGS) $(CONV) mpi_jacobi.c utils.c -o jacobi
	$(CC) $(CFLAGS) mpi_jacobi.c utils.c -o jacobi
clean:
	rm jacobi

# all: jacobi_async

# jacobi_async: mpi_jacobi_async.c utils.c
# 	$(CC) $(CFLAGS) $(CONV) mpi_jacobi_async.c utils.c -o jacobi_async

# clean:
# 	rm jacobi_async

# all: jacobi_sendrcv
#
# jacobi_sendrcv: mpi_jacobi_sendrcv.c utils.c
# 	$(CC) $(CFLAGS) $(CONV) mpi_jacobi_sendrcv.c utils.c -o jacobi_sendrcv
#
# clean:
# 	rm jacobi_sendrcv
