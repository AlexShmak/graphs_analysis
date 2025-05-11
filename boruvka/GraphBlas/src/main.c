#include "LAGraph.h"
#include "LAGraphX.h"
#include "GraphBLAS.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <dirent.h>

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

    GrB_Index *rows = malloc(cedges * sizeof(GrB_Index)); 
    GrB_Index *cols = malloc(cedges * sizeof(GrB_Index));
    int64_t *values = malloc(cedges * sizeof(int64_t));
    int i = 0;

    while(fgets(str, 256, file)) {
        if (str[0] == 'c') {
            continue;
        } else if (str[0] == 'a') {
            char ptype[10];

            if (i > cedges || sscanf(str, "a %ld %ld %ld", &rows[i], &cols[i], &values[i]) != 3) {
                printf("Error parsing 'a' line.\n");
                fclose(file);
                return -1;
            }

            // dimacs indexes start from 1
            rows[i]--;
            cols[i]--;

            i++;
            continue;
        } else {
            printf("Error parsing '%c' symbol.\n", str[0]);
            fclose(file);
            return -1;
        }
    }

    fclose(file);

    GrB_Info info = GrB_Matrix_new(G, GrB_INT64, cnodes, cnodes);
    if (info != GrB_SUCCESS) {
        printf("Error creating matrix.");
        return -1;
    }

    info = GrB_Matrix_build(*G, rows, cols, values, cedges, GrB_PLUS_INT64);
    if (info != GrB_SUCCESS) {
        printf("Error building matrix : %d.", info);
        return -1;
    }

    free(rows);
    free(cols);
    free(values);

    return 0;
}

const int REPEATS = 30;

int run_experiment(char* filename) {
    GrB_Matrix G;
    read_dimacs_file(&G, filename);

    double* time_array = malloc(sizeof(double) * REPEATS);

    for (int i = 0; i < REPEATS; i++) {
        GrB_Matrix result = NULL;

        clock_t begin = clock();
        GrB_Info info = LAGraph_msf(&result, G, true, NULL);
        if (info != GrB_SUCCESS) {
            printf("Error in Boruvka: %d.", info);
            return -1;
        }
        clock_t end = clock();

        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        time_array[i] = time_spent;

        GrB_free(&result);    
    }

    free(time_array);
    GrB_free(&G);

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
    double sem = stddev / sqrt(REPEATS);

    double t_value = 2.042;
    double margin = t_value * sem;

    printf("Mean = %.6fs\n", mean);
    printf("StdDev = %.6fs\n", stddev);
    printf("95%% CI = %.6f ± %.6fs\n", mean, margin);

    return 0;
}

int main(void) {
    GrB_init(GrB_BLOCKING);
    LAGraph_Init(NULL);

    const char *folder_path = "../../graph-utils/exp1_2_dataset";
    DIR *dir = opendir(folder_path);

    if (dir == NULL) {
        perror("opendir failed.");
        return -1;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        printf("Run experiment for: %s...\n", entry->d_name);

        char full_path[4096];
        snprintf(full_path, sizeof(full_path), "%s/%s", folder_path, entry->d_name);

        run_experiment(full_path);
    }

    closedir(dir);

    GrB_finalize();
    return 0;
}
