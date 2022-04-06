
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>
#include "gestione_errori.h"

// dimensioni programma
#define N_BLOCCHI 8
#define N_THREAD_X_BLOCCO 256
#define DIM_INPUT (N_BLOCCHI*N_THREAD_X_BLOCCO)		// dim complessiva intput


// ------------------------------------------------
// funzioni host varie

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

// ------------------------------------------------
// funzioni per il device



/// <summary>
/// Entry point per eseguire codice su device 
/// </summary>
/// <param name="a">puntatore a vettore di interi (allocato su device) di lunghezza DIM_INPUT</param>
/// <param name="b">puntatore a vettore di interi (allocato su device) di lunghezza DIM_INPUT</param>
/// <param name="result">puntatore a vettore di interi (allocato su device) di lunghezza DIM_INPUT</param>
/// <returns>Codice di successo o di errore</returns>
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
	int* aDevice, * bDevice, * resultDevice;

	// Con chiamata equivalente al malloc di C, alloco memoria 
	// su device passando gli indirizzi. Su questi pezzi di memoria
	// posso accedere sia da host che da device.
	HANDLE_ERROR(cudaMalloc((void**)&aDevice, DIM_INPUT * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&bDevice, DIM_INPUT * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&resultDevice, DIM_INPUT * sizeof(int)));


	// ------------------------------------------------
	// copia dei dati sul device

	// per la copia dei dati dalla memoria al device
	HANDLE_ERROR(cudaMemcpy(
		aDevice,	// indirizzo nel device
		a,			// dati da copiare
		DIM_INPUT * sizeof(int),	// dimensione dei dati
		cudaMemcpyHostToDevice	// indico che il trasf. è host -> device
	));

	HANDLE_ERROR(cudaMemcpy(bDevice, b, DIM_INPUT * sizeof(int), cudaMemcpyHostToDevice));


	// ------------------------------------------------
	// esecuzione codice su device

	// NOTA: NON posso applicare HANDLE_ERROR sulla mia funzione, qua dovrò fare altro
	addParallel<<<N_BLOCCHI, N_THREAD_X_BLOCCO>>>(aDevice, bDevice, resultDevice);

	// (forzo la sincronizzazione per essere sicuro di aver terminato) 
	cudaDeviceSynchronize();
	checkKernelError("Errore nell'op. di somma");


	// estrazione del risultato
	// (NOTA: normalmente le memcpy sono punti di sync impliciti...)
	HANDLE_ERROR(cudaMemcpy(result, resultDevice, DIM_INPUT * sizeof(int), cudaMemcpyDeviceToHost));

	// liberazione della memoria sul device
	HANDLE_ERROR(cudaFree(aDevice));
	HANDLE_ERROR(cudaFree(bDevice));
	HANDLE_ERROR(cudaFree(resultDevice));

	printResult(a, b, result);
}