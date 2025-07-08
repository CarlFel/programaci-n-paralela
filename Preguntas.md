# RESPUESTAS

### (i) Parallelize this loop using OpenMP. Are there any data dependencies or shared

## Codigo paralelizado

```c
#pragma omp parallel for
for (k = 1; k <= N; k++) {
    double thread_start = omp_get_wtime(); // Tiempo de inicio del thread
    
    zeta_results[k-1] = Riemann_Zeta(s, k);
    errors[k-1] = fabs(zeta_results[k-1] - exact_value);
    
    double thread_end = omp_get_wtime(); // Tiempo de fin del thread
    int thread_id = omp_get_thread_num();
    thread_times[thread_id] += (thread_end - thread_start) * 1000.0; // tiempo en ms
}
```


## 1. Variables Compartidas

### `zeta_results` y `errors`
Estos arreglos son compartidos entre todos los hilos, ya que se declaran fuera de la región de paralelizacion. Pero, como cada hilo se escribe en un indice unico (´k-1´) no existen condiciones de carrera porque las iteraciones del bucle (`k`) se dividen entre los hilos mediante `#pragma omp parallel for`, garantizando que cada hilo acceda a posiciones de memoria distintas (ej. `zeta_results[0]`, `zeta_results[1]`, etc.).

### `thread_times`
Este arreglo tambíen es compartido, pero cada hilo actualiza solo su propia posicion (´thread_times[thread_id]´), por lo que no habra conflicto

---

### `exact_value`, `s`, `N`
Estas variables son de solo lectura, por lo que no hay riesgo de condicones de carrera.




## 2. Dependencias de Datos

No existe dependencia entre iteraciones, ya que la funcion (´Riemann_Zeta´) es meramente matematica y no tiene dependencia entre las iteraciones, es decir, cada calculo para un ´k´ es independiente

No hay dependencia entre los hilos, ya que los resultados no dependen entre si. Cada ´k´ se procesa de forma independiente.


### (ii) Elaborate on the load balancing of the individual threads.

El balanceo de carga entre los hilos se manejan mediante la directiva de OpenMP ´#pragma omp parallel for´. Esto por el mecanismo que OpenMP divide el bucle ´for (k=1, k<=N; k++)´ entre los hilos disponibles, es decir, cada hilo recibe un conjunto de iteraciones para procesar.

Además, la función (´Riemann_Zeta´) tiene un costo computacional constante para cada ´k´, ya que todos los bucles iteran desde ´1´ hasta ´k´. y como el número de operaciones aumenta con ´k´ y las iteraciones se distribuyen de forma estática por defecto, generan un desequilibrio como lo muestra: 

| Thread | Tiempo (ms) |
|--------|-------------|
| 0      | 0.428       |
| 1      | 0.220       |
| 2      | 0.573       |
| 3      | 1.143       |
| 4      | 1.664       |
| 5      | 2.687       |
| 6      | 2.976       |
| 7      | 3.648       |

Tiempo total de computo: 3.722 ms

Así, de esta forma seguimos con la siguiente implementación (´2_zeta.c´), ajustar el tamaño de cada bloque utilizando ´schedule(static, chunk_size)´, utilizando los chunks se puede controlar explícitamente cuántas iteraciones se asignan a cada thread, evitando que un solo thread reciba iteraciones costosas.

Implementación del chunk en el codigo ´2_zeta.c´:

```c
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
```
Tomando el valor de chunk como: $$ chunk=\frac{1}{\alpha}\frac{N}{p}$$

donde $N$ corresponde al valor de $k=100$, y $p=7$ que son los threads y $\alpha$ es un parámetro empírico, se ajusta su valor observando las pruebas de rendimiento. Para nuestro caso, será $\alpha=1$

´´´ // Configuración de chunks para distribución estática
    const int chunk_size = 14;  ´´´


Tiempos por Thread utilizando chunk=14 (en milisegundos)


| Thread | Tiempo (ms) |
|--------|-------------|
| 0      | 2.610       |
| 1      | 2.558       |
| 2      | 2.549       |
| 3      | 2.604       |
| 4      | 2.626       |
| 5      | 2.552       |
| 6      | 2.586       |
| 7      | 2.161       |

Tiempo total de computo: 2.860 ms

De esta forma, podemos balancear la carga que recibe cada thread y observamos que ahora todos poseen tiempos de computo similares en contraste al codigo inicial ´1_zeta´


### (iii) Discuss different scheduling modes and the corresponding runtimes.

Ahora se discutirá la implementación de diferentes scheduling modes, como el estático, dinámico, guiado y auto.

El estático asigna chunks de iteraciones a los hilos antes de que comience la ejecución por medio de una distribución cíclica por bloques. Ideal para cargas de trabajo uniformes

Dinámico corresponde a que todas las iteraciones se dividen en fragmentos de igual tamaño y se distribuyen entre los hilos en ejecución. Ideal para iteraciones más costosas, por ejemplo con cargas irregulares

El guiado es el que los chunks se dividen en tamaños decrecientes y los batches se distribuyen al igual que el dinámico. Equilibra carga.

El automatico, es el cual el compilador decide la estrategia a utilizar, es poco usado debido a la impredecibilidad. 

Ahora, para poner estos diferentes scheduling a prueba, se diseño el codigo ´3_zeta.c´ donde se encuentran las difentes scheduling para ver los diferentes tiempos de computo. Los resultados fueron: 

| Método                     | Chunk Size | Tiempo (s)  |
|---------------------------|------------|-------------|
| Paralelización estática   | 1          | 0.101016    |
| Paralelización dinámica   | 1          | 0.090235    |
| Paralelización guiada     | —          | 0.092408    |
| Paralelización automática | —          | 0.159267    |


