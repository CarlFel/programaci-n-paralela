#include <stdio.h>
#include <omp.h>

int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

#pragma omp declare reduction(gcd_custom: int: \
    omp_out = gcd(omp_out, omp_in)) \
    initializer(omp_priv = 0)  // Inicializa a 0 (gcd(a, 0) = a)

int main() {
    int numbers[] = {24, 36, 48, 60, 72};
    int n = sizeof(numbers)/sizeof(numbers[0]);
    int result = 0;  // valor neutro para GCD

    #pragma omp parallel for reduction(gcd_custom: result)
    for (int i = 0; i < n; i++) {  // Itera desde 0 
        result = gcd(result, numbers[i]);
    }
    
    printf("GCD: %d\n", result);
    return 0;
}