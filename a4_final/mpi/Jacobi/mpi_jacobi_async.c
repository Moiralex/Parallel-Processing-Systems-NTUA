#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"
#include "utils.h"

// sleep
#include <unistd.h>


// coppied from Jacobi_serial
void Jacobi(double ** u_previous, double ** u_current, int X_min, int X_max, int Y_min, int Y_max) {
	int i,j;
	for (i=X_min;i<X_max;i++)
		for (j=Y_min;j<Y_max;j++)
			u_current[i][j]=(u_previous[i-1][j]+u_previous[i+1][j]+u_previous[i][j-1]+u_previous[i][j+1])/4.0;
}

int main(int argc, char ** argv) {
    int rank,size;
    int global[2],local[2]; //global matrix dimensions and local matrix dimensions (2D-domain, 2D-subdomain)
    int global_padded[2];   //padded global matrix dimensions (if padding is not needed, global_padded=global)
    int grid[2];            //processor grid dimensions
    int i,j,t;
    int global_converged=0,converged=0; //flags for convergence, global and per process
    MPI_Datatype dummy;     //dummy datatype used to align user-defined datatypes in memory
    double omega; 			//relaxation factor - useless for Jacobi

    struct timeval tts,ttf,tcs,tcf;   //Timers: total-> tts,ttf, computation -> tcs,tcf
    double ttotal=0,tcomp=0,total_time,comp_time;

    double ** U, ** u_current, ** u_previous, ** swap; //Global matrix, local current and previous matrices, pointer to swap between current and previous


    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    //----Read 2D-domain dimensions and process grid dimensions from stdin----//

    if (argc!=5) {
        fprintf(stderr,"Usage: mpirun .... ./exec X Y Px Py");
        exit(-1);
    }
    else {
        global[0]=atoi(argv[1]);
        global[1]=atoi(argv[2]);
        grid[0]=atoi(argv[3]);
        grid[1]=atoi(argv[4]);
    }

    //----Create 2D-cartesian communicator----//
	//----Usage of the cartesian communicator is optional----//

    MPI_Comm CART_COMM;         //CART_COMM: the new 2D-cartesian communicator
    int periods[2]={0,0};       //periods={0,0}: the 2D-grid is non-periodic
    int rank_grid[2];           //rank_grid: the position of each process on the new communicator

    MPI_Cart_create(MPI_COMM_WORLD,2,grid,periods,0,&CART_COMM);    //communicator creation
    MPI_Cart_coords(CART_COMM,rank,2,rank_grid);	                //rank mapping on the new communicator

    //----Compute local 2D-subdomain dimensions----//
    //----Test if the 2D-domain can be equally distributed to all processes----//
    //----If not, pad 2D-domain----//

    for (i=0;i<2;i++) {
        if (global[i]%grid[i]==0) {
            local[i]=global[i]/grid[i];
            global_padded[i]=global[i];
        }
        else {
            local[i]=(global[i]/grid[i])+1;
            global_padded[i]=local[i]*grid[i];
        }
    }

	//Initialization of omega
    omega=2.0/(1+sin(3.14/global[0]));

    //----Allocate global 2D-domain and initialize boundary values----//
    //----Rank 0 holds the global 2D-domain----//
    // if (rank==0) {
	if(1){
        U=allocate2d(global_padded[0],global_padded[1]);
        init2d(U,global[0],global[1]);
    }

	// if (rank==0){
	// 	print2d(U,global[0], global[1]);
	// }

    //----Allocate local 2D-subdomains u_current, u_previous----//
    //----Add a row/column on each size for ghost cells----//

    u_previous=allocate2d(local[0]+2,local[1]+2);
    u_current=allocate2d(local[0]+2,local[1]+2);

    //----Distribute global 2D-domain from rank 0 to all processes----//

 	//----Appropriate datatypes are defined here----//
	/*****The usage of datatypes is optional*****/

    //----Datatype definition for the 2D-subdomain on the global matrix----//

    MPI_Datatype global_block;
    MPI_Type_vector(local[0],local[1],global_padded[1],MPI_DOUBLE,&dummy);
    MPI_Type_create_resized(dummy,0,sizeof(double),&global_block);
    MPI_Type_commit(&global_block);

    //----Datatype definition for the 2D-subdomain on the local matrix----//

    MPI_Datatype local_block;
    MPI_Type_vector(local[0],local[1],local[1]+2,MPI_DOUBLE,&dummy);
    MPI_Type_create_resized(dummy,0,sizeof(double),&local_block);
    MPI_Type_commit(&local_block);

    //----Datatype definition for the 2D-subdomain on the local matrix reception----//

    MPI_Datatype receive_block;
    MPI_Type_vector(local[0],local[1],local[1],MPI_DOUBLE,&dummy);
    MPI_Type_create_resized(dummy,0,sizeof(double),&receive_block);
    MPI_Type_commit(&receive_block);


    //----Rank 0 defines positions and counts of local blocks (2D-subdomains) on global matrix----//
    int * scatteroffset, * scattercounts;
    // if (rank==0) {
	if (1){
        scatteroffset=(int*)malloc(size*sizeof(int));
        scattercounts=(int*)malloc(size*sizeof(int));
        for (i=0;i<grid[0];i++)
            for (j=0;j<grid[1];j++) {
                scattercounts[i*grid[1]+j]=1;
                scatteroffset[i*grid[1]+j]=(local[0]*local[1]*grid[1]*i+local[1]*j);
            }
    }


    //----Rank 0 scatters the global matrix----//

    //----Rank 0 scatters the global matrix----//

	//*************TODO*******************//



	/*Fill your code here*/

	/*Make sure u_current and u_previous are
		both initialized*/
    zero2d(u_current,local[0]+2,local[1]+2);
    zero2d(u_previous,local[0]+2,local[1]+2);

    // double **receive_buffer;
    // receive_buffer = allocate2d(local[0], local[1]);
    // zero2d(receive_buffer, local[0], local[1]);

    // if (rank == 0)
    // {
    //     print2d(U, global_padded[0], global_padded[1]);
    // }

    MPI_Scatterv(
        *U, // send buffer , * in order to be tolerated as a 1d array
        scattercounts, // total number of processes / blocks
        scatteroffset, // displacements
        global_block, // send type
        (*u_current+local[1]+3), // receive buffer, * in order to be tolerated as a 1d array
        1, // 1 block per process
        local_block, // receive type
        0, // sender id
        MPI_COMM_WORLD // community
    );


     //************************************//


    if (rank==0)
        free2d(U);

	//----Define datatypes or allocate buffers for message passing----//

	//*************TODO*******************//



	/*Fill your code here*/

    //----Datatype definition row exchange ----//

    MPI_Datatype exchange_row;
    MPI_Type_vector(1,local[1],local[1],MPI_DOUBLE,&dummy);
    MPI_Type_create_resized(dummy,0,sizeof(double),&exchange_row);
    MPI_Type_commit(&exchange_row);

    //----Datatype definition column exchange ----//

    MPI_Datatype exchange_column;
    MPI_Type_vector(local[0],1,local[1]+2,MPI_DOUBLE,&dummy);
    MPI_Type_create_resized(dummy,0,sizeof(double),&exchange_column);
    MPI_Type_commit(&exchange_column);

	//************************************//


    //----Find the 4 neighbors with which a process exchanges messages----//

	//*************TODO*******************//
    int north, south, east, west;

	/*Fill your code here*/

	/*Make sure you handle non-existing
		neighbors appropriately*/
    int north_flag=1;
    int south_flag=1;
    int east_flag=1;
    int west_flag=1;
    MPI_Cart_shift(CART_COMM, 0, 1, &north, &south);
    MPI_Cart_shift(CART_COMM, 1, 1, &west, &east);

    // if current process is the highest one, there is no north
    if (rank_grid[0]==0) north_flag = 0;

    // if current process is the lowest one, there is no south
    if (rank_grid[0]==grid[0]-1) south_flag = 0;

    // if current process is the leftomost one, there is no west
    if (rank_grid[1] == 0) west_flag = 0;

    // if current process is the rightmost one, there is no east
    if (rank_grid[1] == grid[1]-1) east_flag =0;

	//************************************//

    //---Define the iteration ranges per process-----//
	//*************TODO*******************//

    int i_min,i_max,j_min,j_max;

	/*Fill your code here*/

	/*Three types of ranges:
		-internal processes
		-boundary processes
		-boundary processes and padded global array
	*/

    // i limits
    // if both of the above happen
    if (rank_grid[0] == 0 && rank_grid[0] == grid[0]-1)
    {
        i_min = 2;
        // calculate pad (it can be 0)
        int pad = local[0]*grid[0] - global[0];
        i_max = local[0]-pad;
    }
    // upper boundary
    else if (rank_grid[0] == 0)
    {
        i_min = 2;
        i_max = local[0]+1;
    }
    // lower boundary
    else if (rank_grid[0] == grid[0]-1)
    {
        i_min = 1;

        // calculate pad (it can be 0)
        int pad = local[0]*grid[0] - global[0];
        i_max = local[0]-pad;
    }
    // internal
    else
    {
        i_min = 1;
        i_max = local[0]+1;
    }

    // j limits
    // both of the above happen
    if (rank_grid[1] == 0 && rank_grid[1] == grid[1]-1)
    {
        j_min = 2;
        int pad = local[1]*grid[1] - global[1];
        j_max = local[1]-pad;
    }
    // right boundary
    else if (rank_grid[1] == 0)
    {
        j_min = 2;
        j_max = local[1]+1;
    }
    // left boundary
    else if (rank_grid[1] == grid[1]-1)
    {
        j_min = 1;
        int pad = local[1]*grid[1] - global[1];
        j_max = local[1]-pad;
    }
    // internal
    else
    {
        j_min = 1;
        j_max = local[1]+1;
    }

	// start communication

	MPI_Request reqs[8];
	MPI_Status stats[8];
	// reqs=(MPI_Request*)malloc(4*sizeof(MPI_Request));
	// stats=(MPI_Status*)malloc(4*sizeof(MPI_Status));
	int count_neighbours = 0;

	// communicate with south neighbour
	if (south_flag){
		MPI_Isend(
			(*u_current+local[0]*(local[1]+2)+1), // sendbuffer
			1, // sendcount
			exchange_row, // sendtype
			south, // destination
			0, // send tag
			MPI_COMM_WORLD, // community
			&reqs[count_neighbours]//&reqs[count_neighbours]
		);
		count_neighbours += 1;
		MPI_Irecv(
			(*u_current+(local[0]+1)*(local[1]+2)+1), // receive buffer
			1, // receive count
			exchange_row, // receive type
			south, // source
			0, // receive tag
			MPI_COMM_WORLD, // community
			&reqs[count_neighbours]
		);
		count_neighbours+=1;
	}
	// communicate with north neighbor
	if (north_flag){
		MPI_Isend(
			(*u_current+(local[1]+2)+1), // sendbuffer
			1, // sendcount
			exchange_row, // sendtype
			north, // destination
			0, // send tag
			MPI_COMM_WORLD, // community
			&reqs[count_neighbours]
		);
		count_neighbours += 1;
		MPI_Irecv(
			(*u_current+1), // receive buffer
			1, // receive count
			exchange_row, // receive type
			north, // source
			0, // receive tag
			MPI_COMM_WORLD, // community
			&reqs[count_neighbours]
		);
		count_neighbours+=1;
	}
	// communicate with west neighbour
	if (west_flag)
	{
		MPI_Isend(
			(*u_current+(local[1]+2)+1), // sendbuffer
			1, // sendcount
			exchange_column, // sendtype
			west, // destination
			0, // send tag
			MPI_COMM_WORLD, // community
			&reqs[count_neighbours]
		);
		count_neighbours += 1;
		MPI_Irecv(
			(*u_current+(local[1]+2)), // receive buffer
			1, // receive count
			exchange_column, // receive type
			west, // source
			0, // receive tag
			MPI_COMM_WORLD, // community
			&reqs[count_neighbours]
		);
		count_neighbours+=1;
	}
	// communicate with east neighbour
	if (east_flag)
	{
		MPI_Isend(
			(*u_current+(local[1]+2)+(local[1])), // sendbuffer
			1, // sendcount
			exchange_column, // sendtype
			east, // destination
			0, // send tag
			MPI_COMM_WORLD, // community
			&reqs[count_neighbours]
		);
		count_neighbours += 1;
		MPI_Irecv(
			(*u_current+(local[1]+2)+(local[1]+1)), // receive buffer
			1, // receive count
			exchange_column, // receive type
			east, // source
			0, // receive tag
			MPI_COMM_WORLD, // community
			&reqs[count_neighbours]
		);
		count_neighbours+=1;
	}
	MPI_Waitall(count_neighbours, reqs, MPI_STATUS_IGNORE);
	// MPI_Barrier(MPI_COMM_WORLD);
	// end communication

	// init u_previous
	for (i = 0; i < local[0]+2; ++i){
		for (j = 0; j < local[1]+2; ++j){
			u_previous[i][j] = u_current[i][j];
		}
	}

	//************************************//
	// MPI_Request * reqs;
	// MPI_Status *stats;
	// reqs=(MPI_Request*)malloc(4*sizeof(MPI_Request));
	// stats=(MPI_Status*)malloc(4*sizeof(MPI_Status));
	// int count_neighbours = 0;

 	//----Computational core----//
	gettimeofday(&tts, NULL);
    #ifdef TEST_CONV

	for (t=0;t<T && !global_converged;t++) {
	#undef T
    #define T 256
    for (t=0;t<T;t++) {

	#endif
    #ifndef TEST_CONV
    #undef T
    #define T 256
	// #define T 3
    for (t=0;t<T;t++) {
    #endif

	 	//*************TODO*******************//
		/*Fill your code here*/
		/*Compute and Communicate*/
		/*Add appropriate timers for computation*/

		swap=u_previous;
		u_previous=u_current;
		u_current=swap;

        // start computation
        gettimeofday(&tcs,NULL);

		Jacobi(u_previous,u_current,i_min,i_max,j_min,j_max);

		gettimeofday(&tcf,NULL);
        // end computation
        tcomp+=(tcf.tv_sec-tcs.tv_sec)+(tcf.tv_usec-tcs.tv_usec)*0.000001;

        // start communication

		MPI_Request reqs[8];
		MPI_Status stats[8];
		// reqs=(MPI_Request*)malloc(4*sizeof(MPI_Request));
		// stats=(MPI_Status*)malloc(4*sizeof(MPI_Status));
		int count_neighbours = 0;

        // communicate with south neighbour
        if (south_flag){
            MPI_Isend(
				(*u_current+local[0]*(local[1]+2)+1), // sendbuffer
                1, // sendcount
                exchange_row, // sendtype
                south, // destination
                0, // send tag
				MPI_COMM_WORLD, // community
				&reqs[count_neighbours]
			);
			count_neighbours += 1;
			MPI_Irecv(
				(*u_current+(local[0]+1)*(local[1]+2)+1), // receive buffer
                1, // receive count
                exchange_row, // receive type
                south, // source
                0, // receive tag
                MPI_COMM_WORLD, // community
				&reqs[count_neighbours]
			);
			count_neighbours+=1;
        }
        // communicate with north neighbor
        if (north_flag){
			MPI_Isend(
				(*u_current+(local[1]+2)+1), // sendbuffer
                1, // sendcount
                exchange_row, // sendtype
                north, // destination
                0, // send tag
				MPI_COMM_WORLD, // community
				&reqs[count_neighbours]
			);
			count_neighbours += 1;
			MPI_Irecv(
				(*u_current+1), // receive buffer
                1, // receive count
                exchange_row, // receive type
                north, // source
                0, // receive tag
                MPI_COMM_WORLD, // community
				&reqs[count_neighbours]
			);
			count_neighbours+=1;
        }
        // communicate with west neighbour
        if (west_flag)
        {
			MPI_Isend(
				(*u_current+(local[1]+2)+1), // sendbuffer
                1, // sendcount
                exchange_column, // sendtype
                west, // destination
                0, // send tag
				MPI_COMM_WORLD, // community
				&reqs[count_neighbours]
			);
			count_neighbours += 1;
			MPI_Irecv(
				(*u_current+(local[1]+2)), // receive buffer
                1, // receive count
                exchange_column, // receive type
                west, // source
                0, // receive tag
                MPI_COMM_WORLD, // community
				&reqs[count_neighbours]
			);
			count_neighbours+=1;
        }
        // communicate with east neighbour
        if (east_flag)
        {
			MPI_Isend(
				(*u_current+(local[1]+2)+(local[1])), // sendbuffer
                1, // sendcount
                exchange_column, // sendtype
                east, // destination
                0, // send tag
				MPI_COMM_WORLD, // community
				&reqs[count_neighbours]
			);
			count_neighbours += 1;
			MPI_Irecv(
				(*u_current+(local[1]+2)+(local[1]+1)), // receive buffer
                1, // receive count
                exchange_column, // receive type
                east, // source
                0, // receive tag
                MPI_COMM_WORLD, // community
				&reqs[count_neighbours]
			);
			count_neighbours+=1;
        }
		// MPI_Barrier(MPI_COMM_WORLD);
        // end communication

		#ifdef TEST_CONV
        if (t%C==0) {
			//*************TODO**************//
			/*Test convergence*/
            // ? not sure ?

			// gettimeofday(&tcs,NULL);
            converged=converge(u_previous,u_current,i_min,i_max-1,j_min,j_max-1);
			// gettimeofday(&tcf,NULL);
			// tcomp+=(tcf.tv_sec-tcs.tv_sec)+(tcf.tv_usec-tcs.tv_usec)*0.000001;

			MPI_Allreduce(&converged,&global_converged,1,MPI_INT,MPI_LAND,MPI_COMM_WORLD);
			//MPI_Barrier(MPI_COMM_WORLD);
		}
		#endif
		MPI_Waitall(count_neighbours, reqs, MPI_STATUS_IGNORE);

		//************************************//

    }
    gettimeofday(&ttf,NULL);

    ttotal=(ttf.tv_sec-tts.tv_sec)+(ttf.tv_usec-tts.tv_usec)*0.000001;

    MPI_Reduce(&ttotal,&total_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&tcomp,&comp_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);



    //----Rank 0 gathers local matrices back to the global matrix----//

    if (rank==0) {
            U=allocate2d(global_padded[0],global_padded[1]);
            init2d(U, global_padded[0], global_padded[1]);
    }


	//*************TODO*******************//

	/*Fill your code here*/
    MPI_Gatherv(
        (*u_current+local[1]+3), // send buffer
        1,
        local_block, // send type
        *U, // receive buffer
        scattercounts,
        scatteroffset,
        global_block, // receive type
        0, // root
        MPI_COMM_WORLD // community
    );

    // if (rank==0)
    // {
    //     print2d(U, global_padded[0],global_padded[1]);
    // }
     //************************************//



	//----Printing results----//

	//**************TODO: Change "Jacobi" to "GaussSeidelSOR" or "RedBlackSOR" for appropriate printing****************//
    if (rank==0) {
        printf("Jacobi X %d Y %d Px %d Py %d Iter %d ComputationTime %lf TotalTime %lf midpoint %lf\n",global[0],global[1],grid[0],grid[1],t,comp_time,total_time,U[global[0]/2][global[1]/2]);

        #ifdef PRINT_RESULTS
        char * s=malloc(50*sizeof(char));
        sprintf(s,"outdir/resJacobiMPI_%dx%d_%dx%d",global[0],global[1],grid[0],grid[1]);
        fprint2d(s,U,global[0],global[1]);
        free(s);
        #endif

    }
    MPI_Finalize();
    return 0;
}
