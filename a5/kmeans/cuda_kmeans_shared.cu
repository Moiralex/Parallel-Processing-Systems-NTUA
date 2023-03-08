#include <stdio.h>
#include <stdlib.h>

#include "kmeans.h"
#include "alloc.h"
#include "error.h"

#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        // cudaGetErrorString() isn't always very helpful. Look up the error
        // number in the cudaError enum in driver_types.h in the CUDA includes
        // directory for a better explanation.
        error("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}
#endif

__device__ int get_tid(){
	//return 0; /* TODO: copy me from naive version... */
    return threadIdx.x + blockDim.x*blockIdx.x;
}

/* square of Euclid distance between two multi-dimensional points using column-base format */
__host__ __device__ inline static
float euclid_dist_2_transpose(int numCoords,
                    int    numObjs,
                    int    numClusters,
                    float *objects,     // [numCoords][numObjs]
                    float *clusters,    // [numCoords][numClusters]
                    int    objectId,
                    int    clusterId)
{
    int i;
    float ans=0.0;
    float temp;

	/* TODO: Calculate the euclid_dist of elem=objectId of objects from elem=clusterId from clusters*/
    for(i=0; i<numCoords; i++) {
        temp = objects[numObjs*i+objectId]-clusters[numClusters*i+clusterId];
        ans += temp*temp;
        //ans += (objects[numObjs*i+objectId]-clusters[numClusters*i+clusterId])*(objects[numObjs*i+objectId]-clusters[numClusters*i+clusterId]);
    }
    return(ans);
}

__global__ static
void find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
                          float *objects,           //  [numCoords][numObjs]
                          float *deviceClusters,    //  [numCoords][numClusters]
                          int *deviceMembership,          //  [numObjs]
                          float *devdelta)
{
    extern __shared__ char shmem[];

    int *deltas = (int *) shmem;
    float *shmem_clusters = (float *) (shmem+blockDim.x*sizeof(int));

	/* TODO: Copy deviceClusters to shmemClusters so they can be accessed faster. 
		BEWARE: Make sure operations is complete before any thread continues... */

    /* Get the global ID of the thread. */
    int tid = get_tid();

    int n,m;
    int local_tid = threadIdx.x;
    for(n=local_tid; n<numClusters; n+=blockDim.x) {
        for(m=0; m<numCoords; m++) {
            shmem_clusters[numClusters*m+n] = deviceClusters[numClusters*m+n];
        }
    }
    __syncthreads();

	 

	/* TODO: Maybe something is missing here... should all threads run this? */
    if (tid<numObjs) { //was originally 1
        int   index, i;
        float dist, min_dist;

        /* find the cluster id that has min distance to object */
        index = 0;
        /* TODO: call min_dist = euclid_dist_2(...) with correct objectId/clusterId */
        min_dist = euclid_dist_2_transpose(numCoords, numObjs, numClusters, objects, shmem_clusters, tid, 0);

        for (i=1; i<numClusters; i++) {
            /* TODO: call dist = euclid_dist_2(...) with correct objectId/clusterId */
            dist = euclid_dist_2_transpose(numCoords, numObjs, numClusters, objects, shmem_clusters, tid, i);

            /* no need square root */
            if (dist < min_dist) { /* find the min and its array index */
                min_dist = dist;
                index    = i;
            }
        }

        deltas[local_tid] = 0;
        if (deviceMembership[tid] != index) {
        	/* TODO: Maybe something is missing here... is this write safe? */
            //(*devdelta)+= 1.0;
            deltas[local_tid] = 1;
        }

        /* assign the deviceMembership to object objectId */
        deviceMembership[tid] = index;

        /*reduction on deltas first within each thread block then globally*/
        __syncthreads();
        int j = blockDim.x / 2;
        while(j!=0) {
            if(local_tid<j) {
                deltas[local_tid] += deltas[local_tid + j];
            }
            __syncthreads();
            j/=2;
        }
        if (local_tid==0) {
            atomicAdd(devdelta, deltas[0]); //only first thread of each block writes atomically to shared between blocks return value devdelta
        }
    }
}

//
//  ----------------------------------------
//  DATA LAYOUT
//
//  objects         [numObjs][numCoords]
//  clusters        [numClusters][numCoords]
//  dimObjects      [numCoords][numObjs]
//  dimClusters     [numCoords][numClusters]
//  newClusters     [numCoords][numClusters]
//  deviceObjects   [numCoords][numObjs]
//  deviceClusters  [numCoords][numClusters]
//  ----------------------------------------
//
/* return an array of cluster centers of size [numClusters][numCoords]       */            
void kmeans_gpu(	float *objects,      /* in: [numObjs][numCoords] */
		               	int     numCoords,    /* no. features */
		               	int     numObjs,      /* no. objects */
		               	int     numClusters,  /* no. clusters */
		               	float   threshold,    /* % objects change membership */
		               	long    loop_threshold,   /* maximum number of iterations */
		               	int    *membership,   /* out: [numObjs] */
						float * clusters,   /* out: [numClusters][numCoords] */
						int blockSize)  
{
    double timing = wtime(), timing_internal, timer_min = 1e42, timer_max = 0;
    double tmp_timer, C_timer = 0,  G_timer = 0, CG_timer = 0, GC_timer = 0;
	int    loop_iterations = 0; 
    int      i, j, index, loop=0;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    float  delta = 0, *dev_delta_ptr;          /* % of objects change their clusters */
    /* TODO: Copy me from transpose version*/
    float **dimObjects = (float**) calloc_2d(numCoords, numObjs, sizeof(float));
    float **dimClusters = (float**) calloc_2d(numCoords, numClusters, sizeof(float));
    float **newClusters = (float**) calloc_2d(numCoords, numClusters, sizeof(float));
    
    float *deviceObjects;
    float *deviceClusters;
    int *deviceMembership;

    printf("\n|-----------Shared GPU Kmeans------------|\n\n");
    
    /* TODO: Copy me from transpose version*/
	for (i=0; i<numObjs; i++) {
        for(j=0; j<numCoords; j++) {
            dimObjects[j][i] = objects[i*numCoords+j];
        }
    }

    /* pick first numClusters elements of objects[] as initial cluster centers*/
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numClusters; j++) {
            dimClusters[i][j] = dimObjects[i][j];
        }
    }
	
    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL); 
    
    timing = wtime() - timing;
    printf("t_alloc: %lf ms\n\n", 1000*timing);
    timing = wtime();  
    const unsigned int numThreadsPerClusterBlock = (numObjs > blockSize)? blockSize: numObjs;
    const unsigned int numClusterBlocks = (numObjs + numThreadsPerClusterBlock -1) / numThreadsPerClusterBlock; /* TODO: Calculate Grid size, e.g. number of blocks. */

	/*	Define the shared memory needed per block.
    	- BEWARE: We can overrun our shared memory here if there are too many
    	clusters or too many coordinates! 
    	- This can lead to occupancy problems or even inability to run. 
    	- Your exercise implementation is not requested to account for that (e.g. always assume deviceClusters fit in shmemClusters */
    //const unsigned int clusterBlockSharedDataSize = -1;
    /*block size for each thread to store its delta + clusters*coordinates for all clusters*/ 
    const unsigned int clusterBlockSharedDataSize = numThreadsPerClusterBlock*sizeof(int) + numClusters*numCoords*sizeof(float);

    cudaDeviceProp deviceProp;
    int deviceNum;
    cudaGetDevice(&deviceNum);
    cudaGetDeviceProperties(&deviceProp, deviceNum);

    if (clusterBlockSharedDataSize > deviceProp.sharedMemPerBlock) {
        error("Your CUDA hardware has insufficient block shared memory to hold all cluster centroids\n");
    }
           
    checkCuda(cudaMalloc(&deviceObjects, numObjs*numCoords*sizeof(float)));
    checkCuda(cudaMalloc(&deviceClusters, numClusters*numCoords*sizeof(float)));
    checkCuda(cudaMalloc(&deviceMembership, numObjs*sizeof(int)));
    checkCuda(cudaMalloc(&dev_delta_ptr, sizeof(float)));
    
    timing = wtime() - timing;
    printf("t_alloc_gpu: %lf ms\n\n", 1000*timing);
    timing = wtime(); 
    
    checkCuda(cudaMemcpy(deviceObjects, dimObjects[0],
              numObjs*numCoords*sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceMembership, membership,
              numObjs*sizeof(int), cudaMemcpyHostToDevice));
    timing = wtime() - timing;
    printf("t_get_gpu: %lf ms\n\n", 1000*timing);
    timing = wtime();  
    
    do {
    	timing_internal = wtime(); 

		/* GPU part: calculate new memberships */
		        
        /* TODO: Copy clusters to deviceClusters
        checkCuda(cudaMemcpy(...)); */
	tmp_timer = wtime();
        checkCuda(cudaMemcpy(deviceClusters, dimClusters[0], numClusters*numCoords*sizeof(float), cudaMemcpyHostToDevice));
        
        checkCuda(cudaMemset(dev_delta_ptr, 0, sizeof(float)));          

	tmp_timer = wtime() - tmp_timer;
    	CG_timer += tmp_timer; 

	tmp_timer = wtime();
		// printf("Launching find_nearest_cluster Kernel with grid_size = %d, block_size = %d, shared_mem = %d KB\n", numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize/1000);
        // fflush(stdout);
        find_nearest_cluster
            <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numCoords, numObjs, numClusters,
             deviceObjects, deviceClusters, deviceMembership, dev_delta_ptr);

        cudaDeviceSynchronize(); checkLastCudaError();
	tmp_timer = wtime() - tmp_timer;
    	G_timer += tmp_timer;
		//printf("Kernels complete for itter %d, updating data in CPU\n", loop);
		
		/* TODO: Copy deviceMembership to membership
        checkCuda(cudaMemcpy(...)); */
	tmp_timer = wtime();
        checkCuda(cudaMemcpy(membership, deviceMembership, numObjs*sizeof(int), cudaMemcpyDeviceToHost));
    
    	/* TODO: Copy dev_delta_ptr to &delta
        checkCuda(cudaMemcpy(...)); */
        checkCuda(cudaMemcpy(&delta, dev_delta_ptr, sizeof(float), cudaMemcpyDeviceToHost));

	tmp_timer = wtime() - tmp_timer;
    	GC_timer += tmp_timer;
		/* CPU part: Update cluster centers*/
  		
	tmp_timer = wtime();
        for (i=0; i<numObjs; i++) {
            /* find the array index of nestest cluster center */
            index = membership[i];
			
            /* update new cluster centers : sum of objects located within */
            newClusterSize[index]++;
            for (j=0; j<numCoords; j++)
                newClusters[j][index] += objects[i*numCoords + j];
        }
 
        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (newClusterSize[i] > 0)
                    dimClusters[j][i] = newClusters[j][i] / newClusterSize[i];
                newClusters[j][i] = 0.0;   /* set back to 0 */
            }
            newClusterSize[i] = 0;   /* set back to 0 */
        }

        delta /= numObjs;
       	//printf("delta is %f - ", delta);
        loop++; 
	tmp_timer = wtime() - tmp_timer;
    	C_timer += tmp_timer;
        //printf("completed loop %d\n", loop);
	timing_internal = wtime() - timing_internal; 
	if ( timing_internal < timer_min) timer_min = timing_internal; 
	if ( timing_internal > timer_max) timer_max = timing_internal; 
	} while (delta > threshold && loop < loop_threshold);
    
    /*TODO: Update clusters using dimClusters. Be carefull of layout!!! clusters[numClusters][numCoords] vs dimClusters[numCoords][numClusters] */ 
	for (i=0; i<numClusters; i++) {
        for (j=0; j<numCoords; j++) {
            clusters[i*numCoords + j] = dimClusters[j][i];
        }
    }
	
    timing = wtime() - timing;
    printf("nloops = %d  : total = %lf ms\n\t-> t_loop_avg = %lf ms\n\t-> t_loop_min = %lf ms\n\t-> t_loop_max = %lf ms\n\n|-------------------------------------------|\n", 
    	loop, 1000*timing, 1000*timing/loop, 1000*timer_min, 1000*timer_max);

   	printf("t_GPU->CPU = %lf ms\n", 1000*CG_timer); 
   	printf("t_CPU->GPU = %lf ms\n", 1000*GC_timer); 
   	printf("t_GPU = %lf ms\n", 1000*G_timer); 
   	printf("t_CPU = %lf ms\n", 1000*C_timer); 
	char outfile_name[1024] = {0}; 
	sprintf(outfile_name, "Execution_logs/Sz-%ld_Coo-%d_Cl-%d.csv", numObjs*numCoords*sizeof(float)/(1024*1024), numCoords, numClusters);
	FILE* fp = fopen(outfile_name, "a+");
	if(!fp) error("Filename %s did not open succesfully, no logging performed\n", outfile_name); 
	fprintf(fp, "%s,%d,%lf,%lf,%lf\n", "Shmem", blockSize, timing/loop, timer_min, timer_max);
	fclose(fp); 
	
    checkCuda(cudaFree(deviceObjects));
    checkCuda(cudaFree(deviceClusters));
    checkCuda(cudaFree(deviceMembership));

    free(dimObjects[0]);
    free(dimObjects);
    free(dimClusters[0]);
    free(dimClusters);
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return;
}

