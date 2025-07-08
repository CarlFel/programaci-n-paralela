# Preguntas:

**¿Cuál de las siguientes operaciones se puede usar para reducción paralela en OpenMP?**

**(i)** El máximo común divisor de dos o más enteros  
\[
\gcd(a_0, a_1, \dots)
\]

**(ii)** El producto de matrices de dos o más matrices generales de forma \(2 \times 2\)

**(iii)** El producto complejo de dos o más números complejos

**(iv)** El producto de cuaterniones de dos o más cuaterniones

**(v)** La evaluación incremental del promedio de pares de números reales  
\[
\circ: \mathbb{R} \times \mathbb{R} \to \mathbb{R}, \quad (a, b) \mapsto \frac{a + b}{2}
\]


# Operaciones Válidas para Reducción Paralela en OpenMP

Para que una operación sea válida en la reducción paralela en OpenMP, debe cumplir con:

- **Asociatividad**: El orden de evaluación no debe afectar el resultado.
- **Conmutatividad**: combinar los resultados en cualquier orden y dirección (izquierda o derecha, arriba o abajo), simplificando la lógica paralela.

Veremos los casos:

### (i) `gcd(a, b)`

- Ejemplo: `gcd(12, 8) = 4`, `gcd(4, 16) = 4`
- Evaluar como `gcd(12, gcd(8, 16)) = gcd(12, 4) = 4`
- También: `gcd(gcd(12, 8), 16) = gcd(4, 16) = 4`

Por lo tanto, el máximo común divisor es asociativo y conmutativo, lo cual cumple con los requerimientos de reducción paralela.
Aquí parte del código:

``` c
#pragma omp parallel for reduction(gcd_custom: result)
    for (int i = 0; i < n; i++) {  // Itera desde 0 
        result = gcd(result, numbers[i]);
    }

```

### (ii)  Producto de matrices \( A \cdot B \cdot C \)

Debido a que $$ A \times B \neq B \times A $$
implica que el orden en que se multiplican las matrices afecta el resultado final, lo que rompe el requisito de conmutatividad para reducciones en OpenMP.


### (iii) Producto complejo \( (a + bi) \cdot (c + di) \)
- Asociativo y conmutativo.
- por ejemplo:

 \( (1 + i)(2 + i) = (1 \cdot 2 - 1 \cdot 1) + (1 \cdot 1 + 1 \cdot 2)i = (2 - 1) + (1 + 2)i = 1 + 3i \)

De esta forma, el orden del agrupamiento no afecta el resultado y por lo tanto cumpliría para la reducción paralela. Sin embargo, openmp presenta limitaciones tecnicas, ya que no soporta reducciones para estructuras o arrays como ´double(2)´ para números complejos. Sólo soporta reducciones predefinidas para operaciones simples. Pero sí se puede hacer una reducción manual.


### (iv) Producto de cuaterniones. 

Es asociativo:
$$(q_1 \cdot q_2) \cdot q_3 = q_1 \cdot (q_2 \cdot q_3) $$

pero no conmutativo:

$$q_1 \cdot q_2 \neq q_2 \cdot q_1 $$

Sin embargo, aun así se puede usar la reducción, debido a que cumple con la asociatividad, aunque requiere cuidado con el orden de multiplicacion y una estructura adecuada para los cuaterniones de 4 componentes $(w, x, y ,z)$

Aquí parte del código de ´quaternions.c´:

``` c
/ Multiplicación de dos cuaterniones: q = a * b
Quaternion quat_mul(Quaternion a, Quaternion b) {
    Quaternion result;
    result.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    result.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
    result.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
    result.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
    return result;
}

// Reducción personalizada para cuaterniones
#pragma omp declare reduction(quatmul : Quaternion : omp_out = quat_mul(omp_out, omp_in)) initializer(omp_priv = (Quaternion){1,0,0,0})

int main() {
    // Array de 3 cuaterniones
    Quaternion Q[3] = {
        {1, 2, 3, 4},
        {0, 1, 0, 0},
        {0, 0, 1, 0}
    };

    Quaternion result = (Quaternion){1, 0, 0, 0}; // Identidad

    // Reducción paralela 
    #pragma omp parallel for reduction(quatmul:result)
    for (int i = 0; i < 3; i++) {
        result = quat_mul(result, Q[i]);
    }

```

Con un output:

Resultado: -4.00 + -3.00i + 2.00j + 1.00k

### (v) media incremental $\frac{(a+b)}{2}$

si probamos la veracidad del siguiente enunciado:

$$o(o(a,b),c) = o(a, o(b, c)) ?$$

vemos que del lado izquierdo:

$$  o\left(\frac{a + b}{2}, c\right) = \frac{\frac{a + b}{2} + c}{2} = \frac{a + b + 2c}{4}$$

y del lado derecho: 

$$ o\left(a, \frac{b + c}{2}\right) = \frac{a + \frac{b + c}{2}}{2} = \frac{2a + b + c}{4}$$

Entonces: $$\frac{a + b + 2c}{4} \neq \frac{2a + b + c}{4} $$

lo que quiere decir que la operación no es asociativa y no es apta para la reduccion paralela de openmp.
