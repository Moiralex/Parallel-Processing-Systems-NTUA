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
                          float *deviceobjects,           //  [numCoords][numObjs]
                          int *devicenewClusterSize,
                          float *devicenewClusters,
/*                          
                          TODO: If you choose to do (some of) the new centroid calculation here, you will need some extra parameters here (from "update_centroids").
*/                          
                          float *deviceClusters,    //  [numCoords][numClusters]
                          int *deviceMembership,          //  [numObjs]
                          float *devdelta)
{
    extern __shared__ char shmem[];

    int *deltas = (int *) shmem;
    float *shmem_clusters = (float *) (shmem+blockDim.x*sizeof(int));
    float *shmem_newClusters_tmp = (float *) (shmem+blockDim.x*sizeof(int)+numClusters*numCoords*sizeof(float));
    int *shmem_newClusterSize_tmp = (int *) (shmem+blockDim.x*sizeof(int)+2*numClusters*numCoords*sizeof(float));
    // float *shmem_objects_onedim = (float *) (shmem+blockDim.x*sizeof(int)+numClusters*numCoords*sizeof(float));
    // float *

	/* TODO: Copy deviceClusters to shmemClusters so they can be accessed faster. 
		BEWARE: Make sure operations is complete before any thread continues... */

    /* Get the global ID of the thread. */
    int tid = get_tid();

    int n,m;
    int local_tid = threadIdx.x;
    for(n=local_tid; n<numClusters; n+=blockDim.x) {
        for(m=0; m<numCoords; m++) {
            shmem_clusters[numClusters*m+n] = deviceClusters[numClusters*m+n];
            shmem_newClusters_tmp[numClusters*m+n]=0;
        }
        shmem_newClusterSize_tmp[n]=0;
    }
    
    __syncthreads();

	 

	/* TODO: Maybe something is missing here... should all threads run this? */
    if (tid<numObjs) { //was originally 1
        int   index, i;
        float dist, min_dist;

        /* find the cluster id that has min distance to object */
        index = 0;
        /* TODO: call min_dist = euclid_dist_2(...) with correct objectId/clusterId */
        min_dist = euclid_dist_2_transpose(numCoords, numObjs, numClusters, deviceobjects, shmem_clusters, tid, 0);

        for (i=1; i<numClusters; i++) {
            /* TODO: call dist = euclid_dist_2(...) with correct objectId/clusterId */
            dist = euclid_dist_2_transpose(numCoords, numObjs, numClusters, deviceobjects, shmem_clusters, tid, i);

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

        /*update local centers and size*/
        atomicAdd(&(shmem_newClusterSize_tmp[index]), 1);
        for(i=0; i<numCoords; i++)
            atomicAdd(&(shmem_newClusters_tmp[i*numClusters+index]), deviceobjects[i*numObjs+tid]);

        __syncthreads(); /*synchronize for all threads to see the same shared mem*/

        

        /*reduction on deltas first within each thread block then globally*/
        //__syncthreads(); /*already done above*/
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

    for(n=local_tid; n<numClusters*numCoords; n+=blockDim.x) {
            atomicAdd(&(devicenewClusters[n]), shmem_newClusters_tmp[n]);
            if(n<numClusters)
                atomicAdd(&(devicenewClusterSize[n]), shmem_newClusterSize_tmp[n]);
    }

    // __syncthreads();

}

// __global__ static
// void calc_new_clusters(int numCoords,
//                         int numObjs,
//                         int numClusters,
//                         float *deviceObjects, //not transposed objects here
//                         int *deviceMembership,
//                         int *devicenewClusterSize,
//                         float *devicenewClusters
//                         )
// {
//     extern __shared__ char shmem[];

//     float *shobjects = (float *) shmem;
//     int *localnewClusterSize = (int *) (shmem+blockDim.x*numCoords*sizeof(float));
//     float *localnewClusters = (float *) (shmem+blockDim.x*numCoords*sizeof(float)+numClusters*sizeof(int));

//     int tid = get_tid();
//     int local_tid = threadIdx.x;
//     int objId = tid/numClusters;
//     int coordId = tid%numCoords;

//     shobjects[local_tid] = deviceObjects[tid];

// }

__global__ static
void update_centroids(int numCoords,
                          int numClusters,
                          int *devicenewClusterSize,           //  [numClusters]
                          float *devicenewClusters,    //  [numCoords][numClusters]
                          float *deviceClusters)    //  [numCoords][numClusters])
{
    int tid = get_tid();
    int cluster = tid%numClusters;
    int clusterSize;

    if(tid<numClusters*numCoords) {
        clusterSize = devicenewClusterSize[cluster];
        if(clusterSize>0)
            deviceClusters[tid] = devicenewClusters[tid]/clusterSize;
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
	int    loop_iterations = 0; 
    int      i, j, index, loop=0;
    float  delta = 0, *dev_delta_ptr;          /* % of objects change their clusters */
    /* TODO: Copy me from transpose version*/
    float  **dimObjects = (float**) calloc_2d(numCoords, numObjs, sizeof(float)); //calloc_2d(...) -> [numCoords][numObjs]
    float  **dimClusters = (float**) calloc_2d(numCoords, numClusters, sizeof(float));  //calloc_2d(...) -> [numCoords][numClusters]
    float  **newClusters = (float**) calloc_2d(numCoords, numClusters, sizeof(float));  //calloc_2d(...) -> [numCoords][numClusters]

    printf("\n|-----------Full-offload GPU Kmeans------------|\n\n");
    
    /* TODO: Copy me from transpose version*/
	for (i=0; i<numObjs; i++) {
        for(j=0; j<numCoords; j++) {
            dimObjects[j][i] = objects[i*numCoords+j];
        }
    }
    
    float *deviceObjects;
    float *deviceClusters, *devicenewClusters;
    int *deviceMembership;
    int *devicenewClusterSize; /* [numClusters]: no. objects assigned in each new cluster */
    
    /* pick first numClusters elements of objects[] as initial cluster centers*/
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numClusters; j++) {
            dimClusters[i][j] = dimObjects[i][j];
        }
    }
	
    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;
    
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
    const unsigned int clusterBlockSharedDataSize = numThreadsPerClusterBlock*sizeof(int) + 2*numClusters*numCoords*sizeof(float)+ numClusters*sizeof(int); 

    cudaDeviceProp deviceProp;
    int deviceNum;
    cudaGetDevice(&deviceNum);
    cudaGetDeviceProperties(&deviceProp, deviceNum);

    if (clusterBlockSharedDataSize > deviceProp.sharedMemPerBlock) {
        error("Your CUDA hardware has insufficient block shared memory to hold all cluster centroids\n");
    }
           
    checkCuda(cudaMalloc(&deviceObjects, numObjs*numCoords*sizeof(float)));
    checkCuda(cudaMalloc(&deviceClusters, numClusters*numCoords*sizeof(float)));
    checkCuda(cudaMalloc(&devicenewClusters, numClusters*numCoords*sizeof(float)));
    checkCuda(cudaMalloc(&devicenewClusterSize, numClusters*sizeof(int)));
    checkCuda(cudaMalloc(&deviceMembership, numObjs*sizeof(int)));
    checkCuda(cudaMalloc(&dev_delta_ptr, sizeof(float)));
 
    timing = wtime() - timing;
    printf("t_alloc_gpu: %lf ms\n\n", 1000*timing);
    timing = wtime(); 
       
    checkCuda(cudaMemcpy(deviceObjects, dimObjects[0],
              numObjs*numCoords*sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceMembership, membership,
              numObjs*sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceClusters, dimClusters[0],
                  numClusters*numCoords*sizeof(float), cudaMemcpyHostToDevice));
    //checkCuda(cudaMemset(devicenewClusterSize, 0, numClusters*sizeof(int)));
    free(dimObjects[0]);
      
    timing = wtime() - timing;
    printf("t_get_gpu: %lf ms\n\n", 1000*timing);
    timing = wtime();   
    
    do {
        timing_internal = wtime();

        /*initialize delta, newClusterSize and newClusters to 0 on gpu*/
        checkCuda(cudaMemset(dev_delta_ptr, 0, sizeof(float)));
        checkCuda(cudaMemset(devicenewClusterSize, 0, numClusters*sizeof(int)));
        checkCuda(cudaMemset(devicenewClusters, 0, numClusters*numCoords*sizeof(int)));          
		// printf("Launching find_nearest_cluster Kernel with grid_size = %d, block_size = %d, shared_mem = %d KB\n", numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize/1000);
        /* TODO: change invocation if extra parameters needed */
        find_nearest_cluster
            <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numCoords, numObjs, numClusters,
             deviceObjects, devicenewClusterSize, devicenewClusters, deviceClusters, deviceMembership, dev_delta_ptr);
        

        cudaDeviceSynchronize(); checkLastCudaError();
		// printf("Kernels complete for itter %d, updating data in CPU\n", loop);
    
    	/* TODO: Copy dev_delta_ptr to &delta
        checkCuda(cudaMemcpy(...)); */
        checkCuda(cudaMemcpy(&delta, dev_delta_ptr, sizeof(float), cudaMemcpyDeviceToHost));

     	const unsigned int update_centroids_block_sz = (numCoords* numClusters > blockSize) ? blockSize: numCoords* numClusters;  /* TODO: can use different blocksize here if deemed better */
     	const unsigned int update_centroids_dim_sz =  (numCoords*numClusters + update_centroids_block_sz - 1) / update_centroids_block_sz; /* TODO: calculate dim for "update_centroids" and fire it */
     	update_centroids<<< update_centroids_dim_sz, update_centroids_block_sz, 0 >>>
            (numCoords, numClusters, devicenewClusterSize, devicenewClusters, deviceClusters);
        cudaDeviceSynchronize(); checkLastCudaError();   
                       
        delta /= numObjs;
       	// printf("delta is %f - ", delta);
        loop++; 
        // printf("completed loop %d\n", loop);
		timing_internal = wtime() - timing_internal; 
		if ( timing_internal < timer_min) timer_min = timing_internal; 
		if ( timing_internal > timer_max) timer_max = timing_internal; 
	} while (delta > threshold && loop < loop_threshold);
                  	
    checkCuda(cudaMemcpy(membership, deviceMembership,
                 numObjs*sizeof(int), cudaMemcpyDeviceToHost));     
    checkCuda(cudaMemcpy(dimClusters[0], deviceClusters,
                 numClusters*numCoords*sizeof(float), cudaMemcpyDeviceToHost));  
                                   
	for (i=0; i<numClusters; i++) {
		for (j=0; j<numCoords; j++) {
		    clusters[i*numCoords + j] = dimClusters[j][i];
		}
	}
	
    timing = wtime() - timing;
    printf("nloops = %d  : total = %lf ms\n\t-> t_loop_avg = %lf ms\n\t-> t_loop_min = %lf ms\n\t-> t_loop_max = %lf ms\n\n|-------------------------------------------|\n", 
    	loop, 1000*timing, 1000*timing/loop, 1000*timer_min, 1000*timer_max);

	char outfile_name[1024] = {0}; 
	sprintf(outfile_name, "Execution_logs/Sz-%ld_Coo-%d_Cl-%d.csv", numObjs*numCoords*sizeof(float)/(1024*1024), numCoords, numClusters);
	FILE* fp = fopen(outfile_name, "a+");
	if(!fp) error("Filename %s did not open succesfully, no logging performed\n", outfile_name); 
	fprintf(fp, "%s,%d,%lf,%lf,%lf\n", "All_GPU", blockSize, timing/loop, timer_min, timer_max);
	fclose(fp); 
	
    checkCuda(cudaFree(deviceObjects));
    checkCuda(cudaFree(deviceClusters));
    checkCuda(cudaFree(devicenewClusters));
    checkCuda(cudaFree(devicenewClusterSize));
    checkCuda(cudaFree(deviceMembership));

    return;
}

