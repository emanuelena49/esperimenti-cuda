
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// dimensioni programma
#define N_BLOCCHI 2
#define N_THREAD_X_BLOCCO 32
#define DIM_INPUT 64					// dim complessiva intput

void printResult(int* a, int* b, int* result) {
	for (size_t i = 0; i < DIM_INPUT; i++)
	{
		printf("%d+%d=\t%d\n", a[i], b[i], result[i]);
	}
}


void add(int* a, int* b, int* result) {
	for (size_t i = 0; i < DIM_INPUT; i++)
	{
		result[i] = a[i] + b[i];
	}
}

__global__ void addParallel(int* a, int* b, int* result) {

	// Calcolo dell'id (universale) del thread (thread identifier), 
	// lo uso per sapere a quale dato accedere
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadId < DIM_INPUT) {
		result[threadId] = a[threadId] + b[threadId];
	}
}

int main() {

	// ------------------------------------------------
	// dati da elaborare

	int a[DIM_INPUT], b[DIM_INPUT], result[DIM_INPUT];
	for (int i = 0; i < DIM_INPUT; i++) {
		a[i] = -i;
		b[i] = i * i;
	}

	// ------------------------------------------------
	// locazioni di memoria nel device 

	// per ora, li chiedo come puntatori 
	// (poi li alloco come pezzi di memoria sul device)
	int *aDevice, *bDevice, *resultDevice;

	// Con chiamata equivalente al malloc di C, alloco memoria 
	// su device passando gli indirizzi. Su questi pezzi di memoria
	// posso accedere sia da host che da device.
	cudaMalloc((void**)&aDevice, DIM_INPUT * sizeof(int));
	cudaMalloc((void**)&bDevice, DIM_INPUT * sizeof(int));
	cudaMalloc((void**)&resultDevice, DIM_INPUT * sizeof(int));


	// ------------------------------------------------
	// copia dei dati sul device

	// per la copia dei dati dalla memoria al device
	cudaMemcpy(
		aDevice,	// indirizzo nel device
		a,			// dati da copiare
		DIM_INPUT*sizeof(int),	// dimensione dei dati
		cudaMemcpyHostToDevice	// indico che il trasf. è host -> device
		);

	cudaMemcpy(bDevice, b, DIM_INPUT * sizeof(int), cudaMemcpyHostToDevice);


	// ------------------------------------------------
	// esecuzione codice su device
	
	addParallel<<<N_BLOCCHI, N_THREAD_X_BLOCCO>>>(aDevice, bDevice, resultDevice);


	// estrazione del risultato
	cudaMemcpy(result, resultDevice, DIM_INPUT * sizeof(int), cudaMemcpyDeviceToHost);

	// liberazione della memoria sul device
	cudaFree(aDevice);
	cudaFree(bDevice);
	cudaFree(resultDevice);

	printResult(a, b, result);
}