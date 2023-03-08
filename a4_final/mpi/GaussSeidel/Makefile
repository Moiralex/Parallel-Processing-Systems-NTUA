CC=mpicc
CFLAGS=-O3 -lm -Wall
RES=-DPRINT_RESULTS
CONV=-DTEST_CONV

all: gs

gs: mpi_gs.c utils.c
	# $(CC) $(CFLAGS) $(RES) $(CONV) mpi_gs.c utils.c -o gs
	$(CC) $(CFLAGS) mpi_gs.c utils.c -o gs

clean:
	rm gs

