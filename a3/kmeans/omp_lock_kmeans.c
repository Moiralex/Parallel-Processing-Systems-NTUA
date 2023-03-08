#include <stdio.h>
#include <stdlib.h>
#include "kmeans.h"
/*
 * TODO: include openmp header file
 */ 
#include <omp.h>

#include "lock.h"

// square of Euclid distance between two multi-dimensional points
inline static float euclid_dist_2(int    numdims,  /* no. dimensions */
                                 float * coord1,   /* [numdims] */
                                 float * coord2)   /* [numdims] */
{
    int i;
    float ans = 0.0;

    for(i=0; i<numdims; i++)
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);

    return ans;
}

inline static int find_nearest_cluster(int     numClusters, /* no. clusters */
                                       int     numCoords,   /* no. coordinates */
                                       float * object,      /* [numCoords] */
                                       float * clusters)    /* [numClusters][numCoords] */
{
    int index, i;
    float dist, min_dist;

    // find the cluster id that has min distance to object 
    index = 0;
    min_dist = euclid_dist_2(numCoords, object, clusters);

    for(i=1; i<numClusters; i++) {
        dist = euclid_dist_2(numCoords, object, &clusters[i*numCoords]);
        // no need square root 
        if (dist < min_dist) { // find the min and its array index
            min_dist = dist;
            index    = i;
        }
    }
    return index;
}

void kmeans(float * objects,          /* in: [numObjs][numCoords] */
            int     numCoords,        /* no. coordinates */
            int     numObjs,          /* no. objects */
            int     numClusters,      /* no. clusters */
            float   threshold,        /* minimum fraction of objects that change membership */
            long    loop_threshold,   /* maximum number of iterations */
            int   * membership,       /* out: [numObjs] */
            float * clusters)         /* out: [numClusters][numCoords] */
{
    int i, j;
    int index, loop=0;
    double timing = 0;

    float delta;          // fraction of objects whose clusters change in each loop 
    int * newClusterSize; // [numClusters]: no. objects assigned in each new cluster 
    float * newClusters;  // [numClusters][numCoords] 
    int nthreads;         // no. threads 

    nthreads = omp_get_max_threads();
    lock_t *lock; // lock1 -> newClustersSize, lock2 -> newClusters
    lock = lock_init(nthreads);

    printf("OpenMP Kmeans - Lock (%s)\t(number of threads: %d)\n", LOCKNAME, nthreads);

    // initialize membership
    for (i=0; i<numObjs; i++)
        membership[i] = -1;

    // initialize newClusterSize and newClusters to all 0 
    newClusterSize = (typeof(newClusterSize)) calloc(numClusters, sizeof(*newClusterSize));
    newClusters = (typeof(newClusters))  calloc(numClusters * numCoords, sizeof(*newClusters));

    timing = wtime();
    
    do {
        // before each loop, set cluster data to 0
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++)
                newClusters[i*numCoords + j] = 0.0;
            newClusterSize[i] = 0;
        }

        delta = 0.0;

        /* 
         * TODO: Detect parallelizable region and use appropriate OpenMP pragmas
         */
        #pragma omp parallel for \
        private(i,j,index) \
        firstprivate(numObjs,numClusters,numCoords) \
        shared(objects,clusters,membership,newClusters,newClusterSize) \
        schedule(static) reduction(+:delta)

        for (i=0; i<numObjs; i++) {
            // find the array index of nearest cluster center 
            index = find_nearest_cluster(numClusters, numCoords, &objects[i*numCoords], clusters);

            // if membership changes, increase delta by 1 
            if (membership[i] != index)
                delta += 1.0;

            // assign the membership to object i 
            membership[i] = index;

            // update new cluster centers : sum of objects located within 
            lock_acquire(lock);
            newClusterSize[index]++;
            for (j=0; j<numCoords; j++){
                newClusters[index*numCoords + j] += objects[i*numCoords + j];
            }
            lock_release(lock);
        }

        // average the sum and replace old cluster centers with newClusters 
        for (i=0; i<numClusters; i++) {
            if (newClusterSize[i] > 0) {
                for (j=0; j<numCoords; j++) {
                    clusters[i*numCoords + j] = newClusters[i*numCoords + j] / newClusterSize[i];
                }
            }
        }

        // Get fraction of objects whose membership changed during this loop. This is used as a convergence criterion.
        delta /= numObjs;
        
        loop++;
        printf("\r\tcompleted loop %d", loop);
        fflush(stdout);
        // printf("\tcompleted loop %d\n", loop);
        // for (i=0; i<numClusters; i++) {
        //     printf("\tclusters[%ld] = ",i);
        //     for (j=0; j<numCoords; j++)
        //         printf("%6.6f ", clusters[i*numCoords + j]);
        //     printf("\n");
        // }
    } while (delta > threshold && loop < loop_threshold);
    timing = wtime() - timing;
    printf("\n        nloops = %3d   (total = %7.4fs)  (per loop = %7.4f s)\n", loop, timing, timing/loop);

    free(newClusters);
    free(newClusterSize);

    lock_free(lock);
}
