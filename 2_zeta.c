#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>

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
    const int N = 100;
    const double s = 4.0;
    const double exact_value = pow(M_PI, 4)/90.0;
    double *zeta_results, *errors;
    int k;
    double start, end;
    FILE *outfile;
    
    zeta_results = (double*)malloc(N * sizeof(double));
    errors = (double*)malloc(N * sizeof(double));
    
    if (zeta_results == NULL || errors == NULL) {
        fprintf(stderr, "Error de asignación de memoria\n");
        return EXIT_FAILURE;
    }
    
    // Configuración de chunks para distribución estática
    const int chunk_size = 14;  
    
    // Array para almacenar tiempos por thread
    int max_threads = omp_get_max_threads();
    double *thread_times = (double*)calloc(max_threads, sizeof(double));
    
    start = omp_get_wtime();
    
    // Paralelización con distribución estática y chunks definidos
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        double thread_start = omp_get_wtime();
        
        #pragma omp for schedule(static, chunk_size)
        for (k = 1; k <= N; k++) {
            zeta_results[k-1] = Riemann_Zeta(s, k);
            errors[k-1] = fabs(zeta_results[k-1] - exact_value);
        }
        
        double thread_end = omp_get_wtime();
        thread_times[thread_id] = (thread_end - thread_start) * 1000.0; // ms
    }
    
    end = omp_get_wtime();
    double duration = (end - start) * 1000.0;
    
    outfile = fopen("zeta_results_static_chunk.csv", "w");
    if (outfile == NULL) {
        fprintf(stderr, "Error al abrir el archivo\n");
        free(zeta_results);
        free(errors);
        free(thread_times);
        return EXIT_FAILURE;
    }
    
    fprintf(outfile, "k,Zeta(4),Error,Exact_Value\n");
    for (k = 1; k <= N; k++) {
        fprintf(outfile, "%d,%.15f,%.15f,%.15f\n", 
                k, zeta_results[k-1], errors[k-1], exact_value);
    }
    fclose(outfile);
    
    // Mostrar tiempos por thread
    printf("\nTiempos por thread (ms):\n");
    for (int i = 0; i < max_threads; i++) {
        printf("Thread %d: %.3f ms\n", i, thread_times[i]);
    }
    
    printf("\nTiempo total de computo: %.3f ms\n", duration);
    printf("Valor exacto de Zeta(4): %.15f\n", exact_value);
    printf("Resultados guardados en zeta_results_static_chunk.csv\n");
    
    free(zeta_results);
    free(errors);
    free(thread_times);
    
    return EXIT_SUCCESS;
}