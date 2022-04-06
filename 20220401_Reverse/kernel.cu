
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

	bool dynamic=true;

	// printf("Vuoi eseguire l'algoritmo con allocazione dinamica? (con allocazione statica sei limitato ad un vettore di lungh 64) si/no [default: si]");
	// std::string dinamicaSiNo;
	// int res = scanf("%s", &dinamicaSiNo);
	// if (res != EOF && dinamicaSiNo == "no") {
	// 	dynamic = false;
	// 	printf("procedo con allocazione statica (NOTA: la lungh. dell'input è limitata a 64)\n");
	// }
	// else {
	// 	dynamic = true;
	// 	printf("procedo con allocazione dinamica\n");
	// }



	int vLength;
	printf("Inserisci il numero di elementi del vettore di input: ");
	int res = scanf("%d", &vLength);

	if (res == EOF && vLength < 1) {
		printf("Errore: è stato inserito un valore non accettabile "
			"(assicurati di aver inserito un numero intero >= 1)\n");
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


	if (dynamic) {
		dynamicReverse << <1, vLength, vLength * sizeof(int) >> > (vDevice, vLength);
	}
	else {
		staticReverse << <1, vLength >> > (vDevice, vLength);
	}

	cudaDeviceSynchronize();
	checkKernelError("Errore nell'operzione");

	HANDLE_ERROR(cudaMemcpy(vHost, vDevice, vLength * sizeof(int), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(vDevice));

	stampaVettore(vHost, vLength, "v");
}

