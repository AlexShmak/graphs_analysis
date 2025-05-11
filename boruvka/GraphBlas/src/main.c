#include "LAGraph.h"
#include "LAGraphX.h"
#include "GraphBLAS.h"
#include <stdio.h>
#include <string.h>

int read_dimacs_file(GrB_Matrix* G, char* filename) {
    FILE *file = fopen(filename, "r");

    if (file == NULL) {
        printf("Can't open file: %s.", filename);
        return -1;
    }

    char str[100];

    int cnodes, cedges;

    // read header
    while(fgets(str, 100, file)) {
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

    while(fgets(str, 100, file)) {
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

int main(void) {
    GrB_init(GrB_BLOCKING);
    GrB_Matrix G;

    read_dimacs_file(&G, "../../graph-utils/USA-road-d.BAY.gr");
    GrB_finalize();

    return 0;
}
