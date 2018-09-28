/**
  * Maestría en Ciencias - Mención Informática
  * -------------------------------------------
  * Escriba un programa en C que calcule C = n*A + B, en donde A, B, C son 
  * vectores y n una constante escalar y compare los resultados
  * de tiempo de ejecución con los del primer programa
  * 
  * Adaptado de https://www.olcf.ornl.gov/tutorials/cuda-vector-addition/
  * 
  * Presentado por:
  * Zuñiga Rojas, Gabriela
  * Soncco Pimentel, Braulio
  */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

const int k = 5;

double vecAdd(double a, double b, int k) {
	return k * a + b;
}

int main( int argc, char* argv[] )
{

    clock_t tic, toc;
    double secs;

    tic = clock();

    // Size of vectors
    int n = 10000000;
 
    // Host input vectors
    double *h_a;
    double *h_b;
    //Host output vector
    double *h_c;
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
 
    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);
 
    int i;
    // Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        h_a[i] = sin(i)*sin(i);
        h_b[i] = cos(i)*cos(i);
    }
 
    // Execute the kernel

	for( i = 0; i < n; i++) {
		h_c[i] = vecAdd(h_a[i], h_b[i], k);
	}
 
    // Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for(i=0; i<n; i++)
        sum += h_c[i];
    printf("final result: %f\n", sum/n);
 
    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    toc = clock();

    secs = (double)(toc - tic) / CLOCKS_PER_SEC;
    printf("%.16g milisegundos\n", secs * 1000.0);
 
    return 0;
}