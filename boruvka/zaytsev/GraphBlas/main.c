#include "LAGraph.h"
#include "LAGraphX.h"
#include "GraphBLAS.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <dirent.h>
#include <assert.h>
#include <sys/stat.h>

int read_dimacs_file(GrB_Matrix* G, char* filename) {
    FILE *file = fopen(filename, "r");

    if (file == NULL) {
        printf("Can't open file: %s.", filename);
        return -1;
    }

    char str[256];

    int cnodes, cedges;

    // read header
    while(fgets(str, 256, file)) {
        if (str[0] == 'c') {
            continue;
        } else if (str[0] == 'p') {
            char ptype[10];

            if (sscanf(str, "p %s %d %d", ptype, &cnodes, &cedges) != 3) {
                printf("Error parsing 'p' line.\n");
                fclose(file);
                return -1;
            }

            break;
        } else {
            printf("Error parsing '%c' symbol.\n", str[0]);
            fclose(file);
            return -1;
        }
    }

    GrB_Index *u = malloc(cedges * sizeof(GrB_Index)); 
    GrB_Index *v = malloc(cedges * sizeof(GrB_Index));
    int64_t *weight = malloc(cedges * sizeof(int64_t));
    int i = 0;

    while(fgets(str, 256, file)) {
        if (str[0] == 'a') {
            if (i >= cedges || sscanf(str, "a %" SCNu64 " %" SCNu64 " %" SCNd64, &u[i], &v[i], &weight[i]) != 3) {
                printf("Error parsing 'a' line\n");
                free(u);
                free(v);
                free(weight);
                fclose(file);
                return -1;
            }

            // dimacs indexes start from 1
            u[i]--;
            v[i]--;

            i++;
            continue;
        }
    }

    fclose(file);

    GrB_Info info = GrB_Matrix_new(G, GrB_INT64, cnodes, cnodes);
    if (info != GrB_SUCCESS) {
        printf("Error creating matrix.");
        return -1;
    }

    info = GrB_Matrix_build(*G, u, v, weight, i, GrB_PLUS_INT64);
    if (info != GrB_SUCCESS) {
        printf("Error building matrix : %d.", info);
        return -1;
    }

    free(u);
    free(v);
    free(weight);

    return 0;
}

const int REPEATS = 100;

void warm_up(char* filename) {
    GrB_Matrix G;
    GrB_Info info;
    
    printf("Starting warm-up...\n");
    
    int res = read_dimacs_file(&G, filename);
    assert(res == 0);
    
    GrB_Matrix result = NULL;
    info = LAGraph_msf(&result, G, true, NULL);
    
    if (info != GrB_SUCCESS) {
        printf("Warning: Warm-up failed with error %d. Continuing anyway.\n", info);
    } else {
        printf("Warm-up completed successfully.\n");
    }
    
    GrB_free(&result);
    GrB_Matrix_free(&G);
}

int run_experiment(char* filename, FILE* logger) {
    GrB_Matrix G;
    read_dimacs_file(&G, filename);

    double* time_array = malloc(sizeof(double) * REPEATS);

    for (int i = 0; i < REPEATS; i++) {
        GrB_Matrix result = NULL;

        struct timespec begin, end;
        clock_gettime(CLOCK_MONOTONIC, &begin);
        GrB_Info info = LAGraph_msf(&result, G, true, NULL);
        clock_gettime(CLOCK_MONOTONIC, &end);

        double time_spent = (end.tv_sec - begin.tv_sec) + 
                            (end.tv_nsec - begin.tv_nsec) / 1e9;

        time_array[i] = time_spent;

        if (info != GrB_SUCCESS) {
            fprintf(logger, "Error in Boruvka: %d.", info);
            GrB_Matrix_free(&G);
            return -1;
        }

        GrB_Matrix_free(&result);    
    }

    GrB_Matrix_free(&G);

    // --- Statistics

    double sum = 0.0;

    for (int i = 0; i < REPEATS; i++) 
        sum += time_array[i];

    double mean = sum / REPEATS;
    double variance = 0.0;

    for (int i = 0; i < REPEATS; i++) {
        double diff = time_array[i] - mean;
        variance += diff * diff;
    }

    variance /= (REPEATS - 1);

    double stddev = sqrt(variance);
    double z = 1.96;
    double margin = z * (stddev / sqrt(REPEATS));

    fprintf(logger, "Mean = %.6fs\n", mean);
    fprintf(logger, "StdDev = %.6fs\n", stddev);
    fprintf(logger, "95%% CI = ±%.6fs\n", margin);

    free(time_array);
    return 0;
}

void run_experiments_on_dataset(char* foldername) {
    const char *folder_path = foldername;
    DIR *dir = opendir(folder_path);

    if (dir == NULL) {
        perror("opendir failed.");
        return;
    }

    FILE *logger = fopen("logs.txt", "a");
    if (logger == NULL) {
        perror("fopen failed.");
        closedir(dir);
        return;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        char full_path[4096];
        snprintf(full_path, sizeof(full_path), "%s/%s", folder_path, entry->d_name);

        struct stat st;
        if (stat(full_path, &st) == -1) {
            perror("stat failed");
            continue;
        }

        if (!S_ISREG(st.st_mode)) {
            // Not a regular file, skip
            continue;
        }

        printf("Run experiment for: %s\n", entry->d_name);
        fprintf(logger, "Run experiment for: %s\n" ,entry->d_name);

        run_experiment(full_path, logger);
        fflush(logger);
    }

    fclose(logger);
    closedir(dir);
}

void run_tests() {
    GrB_Matrix G;
    read_dimacs_file(&G, "../graph-utils/test-graph.gr");

    // verify matrix size
    GrB_Index nrows, ncols;
    GrB_Matrix_nrows(&nrows, G);
    GrB_Matrix_ncols(&ncols, G);
    assert(nrows == 5);
    assert(ncols == 5);
    
    // verify edges count
    GrB_Index nvals;
    GrB_Matrix_nvals(&nvals, G);
    assert(nvals == 10);

    double value1, value2 = 0;
    GrB_Matrix_extractElement_FP64(&value1, G, 3, 4);
    GrB_Matrix_extractElement_FP64(&value2, G, 4, 3);

    assert(value1 == 12);
    assert(value2 == 0);
}

int main(void) {
    GrB_init(GrB_BLOCKING);
    LAGraph_Init(NULL);
    run_tests();
    // warm_up("../dataset/LKS.gr");

    // run_experiments_on_dataset("../dataset");
    run_experiments_on_dataset("../dataset/ext-dataset-exp1");
    run_experiments_on_dataset("../dataset/ext-dataset-exp2");

    GrB_finalize();
    return 0;
}
