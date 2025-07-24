### 1- 0.5 $T_{sec}$ y 0.5 $T_{par}$. $S_{max}$ en $p$ procesadores


Utilizando la Ley de Amdahl:

$$S(p)=\frac{1}{f+\frac{1-f}{p}}$$

Haciendola tender a infinito los procesadores:

$$\lim_{p \rightarrow \infty} S(p)= \lim_{p \rightarrow \infty} \frac{1}{f + \frac{1-f}{p}} = \frac{1}{f} = \frac{1}{0.5}=2$$


### 2- Escalabilidad débil y fuerte

Escalabilidad débil corresponde a variar el tamaño del problema mantiendo constante el número de procesadores. Es decir, vemos como se comporta la eficiencia $S$ a medida que variamos el tamaño del problema con $p$ constante. Se utiliza cuando se requiere respuesta rápida. Un ejemplo de uso podrían ser las simulaciones a gran escala, como modelado climático 

Escalabilidad fuerte corresponde al caso inverso, es decir, manteniendo el tamaño del problema constante y haciendo variar el número de procesadores. Se analiza como se comporta la eficiencia $S$ para estas condiciones, de esta forma, tiene por objetivo resolver el mismo problema pero más rápido.

### 3- $T_{par}$ crece cuadraticamente en p por un aumento en el n° de datos, $T_{sec}$ constante. ¿fraccion de $T_{sec}$ para asegurar $S$ de 50 en 150 $p$ bajo escalabilidad débil.

Cuando tenemos el número de datos variado bajo escalabilidad débil, utilizamos la Ley de Gustafson:


$$S(p)=p + f(1-p)$$

por lo tanto: $$ 50 = 150 + f(1-150) \Rightarrow \ -100 = -149f \ \Rightarrow \ f= \frac{100}{149}= 0 ,671$$


### 4- paralelizable?

```c
for (int i= 0: i<N; i++){
    A[i]= A[i-2] + A[i+2]
}
```

Por lo que se puede observar, el código presenta dependencia de datos entre las iteraciones del ciclo:

- en cada iteracion, el valor de A[i] depende de los valores A[i-2] anteriores y los siguientes A[i+2]

De esta forma, no es paralelizable directamente.
### 5 - Errores del codigo:


```c
for (int i = 0; i < N; i++){
__m256 X = _mm256_load_ps(x);
__m256 Y = _mm256_load_ps(y);
__m256 Z = _mm256_mul_ps(X, Y);
_mm256_store_ps(z, Z);
}

```

- Error en falta de incremento de los punteros $x$, $y$ y $z$

- Error en i++, ya que no muestra la cantidad de elementos de tipo float que caben en un registro AVX __m256. 

Código arreglado:


```c
for (int i = 0; i < N; i+= 8){
__m256 X = _mm256_load_ps(x+i);
__m256 Y = _mm256_load_ps(y+i);
__m256 Z = _mm256_mul_ps(X, Y);
_mm256_store_ps(z+i, Z);
}

```



### 6- Estrategias de paralelización:

- Paralelismo de datos: Muy factible para el problema de la simulación de la ecuación de onda. Consideramos una malla de $NxN$ donde particionamos esta data para procesarla en diferentes unidades de procesamiento. Podriamos utlizar OpenMP (hilos compartidos) o MPI (procesos distribuidos)


- Paralelismo de tareas: es medio factible, ya que en este caso necesitariamos asignar diferentes tareas a procesar, utilizando multiprocessing y condiciones de borde para descomponer el problema. Por ejemplo, mientras una parte del código calcula la evolución de la onda, otra puede calcular energía total, guardar datos o visualizar.


- Paralelismo de modelo: baja factibilidad. Debido a que necesitariamos particionar el modelo. Podriamos utilizarlo considerando simular la ecuación de onda para diferentes velocidades $v$ o en diferentes dominios espaciales. Así, cada simulación de se ejecuta de forma independiente en paralelo