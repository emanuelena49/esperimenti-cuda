
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include "../20220325_GestioneErrori/gestione_errori.h"
#include "../20220325_GestioneErrori/manipolazioneVettori.h"


/// <summary>
/// Reverse di un array di una dimensione n sfruttando n/2 thread paralleli
/// </summary>
/// <param name="v"></param>
/// <param name="vSize"></param>
/// <returns></returns>
__global__ void staticReverse(int* v, int vLength) {

	// dichiarazione di memoria shared, sarà visibile da tutti i thread del blocco
	__shared__ int s[64];

	int threadId = threadIdx.x;
	int threadIdReciproce = vLength - threadId - 1;

	// copia da memoria globale a memoria shared
	s[threadId] = v[threadId];

	// barrier (non sarebbe necessaria in questo caso)
	__syncthreads();

	v[threadId] = s[threadIdReciproce];
}

__global__ void dynamicReverse(int* v, int vLength) {

	// dichiarazione memoria condivsa ma dinamica
	extern __shared__ int s[];

	int threadId = threadIdx.x;
	int threadIdReciproce = vLength - threadId - 1;

	// copia da memoria globale a memoria shared
	s[threadId] = v[threadId];

	// barrier (non sarebbe necessaria in questo caso)
	__syncthreads();

	v[threadId] = s[threadIdReciproce];
}


int main() {

	int vLength;
	printf("Inserisci il numero di elementi del vettore di input: ");
	int res = scanf("%d", &vLength);

	if (res == EOF) {
		return -1;
	}

	if (res < 1) {
		return -1;
	}

	int *vHost;
	vHost = (int*) malloc(vLength * sizeof(int));

	for (size_t i = 0; i < vLength; i++)
	{
		vHost[i] = i;
	}

	int* vDevice;
	HANDLE_ERROR(cudaMalloc((void**)&vDevice, vLength * sizeof(int)));
	HANDLE_ERROR(cudaMemcpy(vDevice, vHost, vLength * sizeof(int), cudaMemcpyHostToDevice));


	staticReverse<<<1, vLength>>>(vDevice, vLength);

	cudaDeviceSynchronize();
	checkKernelError("Errore nell'operzione");

	HANDLE_ERROR(cudaMemcpy(vHost, vDevice, vLength * sizeof(int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(vDevice));

	stampaVettore(vHost, vLength, "v");
}

