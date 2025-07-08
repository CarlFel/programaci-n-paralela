#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <string.h>

double Riemann_Zeta(double s, int k) {
    double result = 0.0;
    int i, j;
    
    for (i = 1; i <= k; i++) {
        for (j = 1; j <= k; j++) {
            result += (2*(i%2)-1)/pow(i+j, s);
        }
    }
    return result * pow(2, s);
}

int main() {
    const int N = 500;
    const double s = 4.0;
    const double exact_value = pow(M_PI, 4)/90.0;
    double *zeta_results, *errors;
    int k;
    double start, end;
    const int chunk_size = 1;

    // Asignar memoria para resultados y errores
    zeta_results = (double*)malloc(N * sizeof(double));
    errors = (double*)malloc(N * sizeof(double));
    
    if (zeta_results == NULL || errors == NULL) {
        fprintf(stderr, "Error de asignación de memoria\n");
        return EXIT_FAILURE;
    }
    // Scheduling Estático
    start = omp_get_wtime();
    #pragma omp parallel for schedule(static, chunk_size)
    for (k = 1; k <= N; k++) {
        zeta_results[k-1] = Riemann_Zeta(s, k);
        errors[k-1] = fabs(zeta_results[k-1] - exact_value);
    }
    end = omp_get_wtime();
    printf("Paralelización estática:\n");
    printf("         chunk size: %d\n", chunk_size);
    printf("         t: %.6f [s]\n", end - start);

     // Scheduling Dinámico
     start = omp_get_wtime();
     #pragma omp parallel for schedule(dynamic, chunk_size)
     for (k = 1; k <= N; k++) {
         zeta_results[k-1] = Riemann_Zeta(s, k);
         errors[k-1] = fabs(zeta_results[k-1] - exact_value);
     }
     end = omp_get_wtime();
     printf("Paralelización dinámica:\n");
     printf("         chunk size: %d\n", chunk_size);
     printf("         t: %.6f [s]\n", end - start);

    // Scheduling Guiado

    start = omp_get_wtime();
    #pragma omp parallel for schedule(guided)
    for (k = 1; k <= N; k++) {
        zeta_results[k-1] = Riemann_Zeta(s, k);
        errors[k-1] = fabs(zeta_results[k-1] - exact_value);
    }
    end = omp_get_wtime();
    printf("Paralelización guiada:\n");
    printf("         t: %.6f [s]\n", end - start);

    // Scheduling Automático

    start = omp_get_wtime();
    #pragma omp parallel for schedule(auto)
    for (k = 1; k <= N; k++) {
        zeta_results[k-1] = Riemann_Zeta(s, k);
        errors[k-1] = fabs(zeta_results[k-1] - exact_value);
    }
    end = omp_get_wtime();
    printf("Paralelización automática:\n");
    printf("         t: %.6f [s]\n", end - start);

    // Liberar memoria
    free(zeta_results);
    free(errors);

    return EXIT_SUCCESS;
}