#include <stdio.h>
#include <omp.h>

// Estructura para cuaterniones: q = w + xi + yj + zk
typedef struct {
    double w, x, y, z;
} Quaternion;

// Multiplicación de dos cuaterniones: q = a * b
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

    printf("Resultado: %.2f + %.2fi + %.2fj + %.2fk\n", result.w, result.x, result.y, result.z);
    return 0;
}
