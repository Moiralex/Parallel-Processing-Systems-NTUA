CC=mpicc
CFLAGS=-O3 -lm -Wall
RES=-DPRINT_RESULTS
CONV=-DTEST_CONV

all: redBlack

redBlack: mpi_redBlack.c utils.c
	# $(CC) $(CFLAGS) $(RES) $(CONV) mpi_redBlack.c utils.c -o redBlack
	$(CC) $(CFLAGS) mpi_redBlack.c utils.c -o redBlack

clean:
	rm redBlack

