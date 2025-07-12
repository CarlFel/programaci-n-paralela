#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N 50
#define MAX_ITER 10000
#define TOLERANCE 1e-6

// leer archivo csv de malla cargada y mostrar error en caso de problema
void read_csv(double grid[N][N], const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fscanf(file, "%lf,", &grid[i][j]) != 1) {
                fprintf(stderr, "Error reading file at (%d,%d)\n", i, j);
                fclose(file);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }
    fclose(file);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Crear una topología cartesiana 2D
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int periods[2] = {0, 0}; // No periódico
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    // Obtener coordenadas del proceso en la malla
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    // Calcular tamaño del bloque local
    int local_rows = N / dims[0];
    int local_cols = N / dims[1];
    int remainder_rows = N % dims[0];
    int remainder_cols = N % dims[1];

    // Ajustar para división 
    if (coords[0] < remainder_rows) local_rows++;
    if (coords[1] < remainder_cols) local_cols++;

    // Asignar memoria para el bloque local
    double (*local_data)[local_cols] = malloc(local_rows * local_cols * sizeof(double));
    double (*new_data)[local_cols] = malloc(local_rows * local_cols * sizeof(double));
    double (*rho)[local_cols] = malloc(local_rows * local_cols * sizeof(double));

    // Proceso 0 lee los datos y distribuye
    if (rank == 0) {
        double global_data[N][N];
        read_csv(global_data, "distribucion_carga1.csv");

        // Distribuir datos a todos los procesos
        int counts[size], displs[size];
        for (int proc = 0; proc < size; proc++) {
            int proc_coords[2];
            MPI_Cart_coords(cart_comm, proc, 2, proc_coords);

            int proc_rows = N / dims[0];
            int proc_cols = N / dims[1];
            if (proc_coords[0] < remainder_rows) proc_rows++;
            if (proc_coords[1] < remainder_cols) proc_cols++;

            counts[proc] = proc_rows * proc_cols;
            displs[proc] = 0; // 
        }

        MPI_Scatter(global_data, local_rows*local_cols, MPI_DOUBLE,
                   local_data, local_rows*local_cols, MPI_DOUBLE, 0, cart_comm);
    } else {
        MPI_Scatter(NULL, local_rows*local_cols, MPI_DOUBLE,
                   local_data, local_rows*local_cols, MPI_DOUBLE, 0, cart_comm);
    }

    // Inicializar rho 
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < local_cols; j++) {
            rho[i][j] = 0.1; // Valor de ejemplo
        }
    }

    // tipos derivados para comunicación de bordes
    MPI_Datatype column_type;
    MPI_Type_vector(local_rows, 1, local_cols, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    // Iteración de Jacobi
    double global_diff;
    int iter = 0;
    do {
        double local_diff = 0.0;

        // Comunicar bordes con vecinos
        // Enviar/recepcionar columnas izquierda/derecha
        int left, right, up, down;
        MPI_Cart_shift(cart_comm, 1, 1, &left, &right);
        MPI_Cart_shift(cart_comm, 0, 1, &up, &down);

        // Comunicación de columnas fantasmas
        if (right != MPI_PROC_NULL) {
            MPI_Send(&local_data[0][local_cols-1], 1, column_type, right, 0, cart_comm);
        }
        if (left != MPI_PROC_NULL) {
            MPI_Recv(&local_data[0][-1], 1, column_type, left, 0, cart_comm, MPI_STATUS_IGNORE);
        }

        if (left != MPI_PROC_NULL) {
            MPI_Send(&local_data[0][0], 1, column_type, left, 1, cart_comm);
        }
        if (right != MPI_PROC_NULL) {
            MPI_Recv(&local_data[0][local_cols], 1, column_type, right, 1, cart_comm, MPI_STATUS_IGNORE);
        }

        // Aplicar fórmula de diferencias finitas
        for (int i = 1; i < local_rows-1; i++) {
            for (int j = 1; j < local_cols-1; j++) {
                new_data[i][j] = (local_data[i+1][j] + local_data[i-1][j]) / 4.0 +
                                (local_data[i][j+1] + local_data[i][j-1]) / 4.0 -
                                rho[i][j] / 4.0;
                
                local_diff += fabs(new_data[i][j] - local_data[i][j]);
            }
        }

        // Actualizar datos
        for (int i = 1; i < local_rows-1; i++) {
            for (int j = 1; j < local_cols-1; j++) {
                local_data[i][j] = new_data[i][j];
            }
        }

        // Reducir la diferencia global
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
        global_diff /= (N * N);

        iter++;
        if (rank == 0 && iter % 100 == 0) {
            printf("Iteration %d, diff = %g\n", iter, global_diff);
        }
    } while (iter < MAX_ITER && global_diff > TOLERANCE);

    // Recopilar resultados en el proceso 0
    if (rank == 0) {
        double result[N][N];
        MPI_Gather(local_data, local_rows*local_cols, MPI_DOUBLE,
                  result, local_rows*local_cols, MPI_DOUBLE, 0, cart_comm);

        // Guardar resultado (simplificado)
        FILE *out = fopen("poisson_solution.csv", "w");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fprintf(out, "%.6f%s", result[i][j], (j == N-1) ? "\n" : ",");
            }
        }
        fclose(out);
    } else {
        MPI_Gather(local_data, local_rows*local_cols, MPI_DOUBLE,
                  NULL, 0, MPI_DOUBLE, 0, cart_comm);
    }

    // Liberar recursos
    MPI_Type_free(&column_type);
    free(local_data);
    free(new_data);
    free(rho);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();

    return 0;
}