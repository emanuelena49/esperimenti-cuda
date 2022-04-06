#ifndef GESTIONE_ERRORI
#define GESTIONE_ERRORI

#include "cuda_runtime.h"
#include <stdio.h>
#include <cstdlib>


/// <summary>
/// Funzione per la gestione d'errore. Nel caso in cui il codice d'output passato 
/// sia un errore, stampa a console un messaggio d'errore (indicando file e riga)
/// </summary>
/// <param name="cudaFunctionOutput">
///	L'output di una funzione CUDA, può essere un codice d'errore o uno di successo
/// </param>
/// <param name="file">Il file dove è accaduto l'errore</param>
/// <param name="line">La linea di codice che ha causato l'errore</param>
static void HandleError(cudaError_t cudaFunctionOutput, const char* file, int line) {
	if (cudaFunctionOutput != cudaSuccess) {
		fprintf(stderr, "%s: %s in %s at line %d\n", cudaGetErrorName(cudaFunctionOutput),
			cudaGetErrorString(cudaFunctionOutput), file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))


// NOTA: poi potrei avere macro diverse per dev e prod 
// (rimuovendo controlli, introducendo gest. degli errori, ecc.)

// NOTA: la macro non può essere usata per l'esecuzione del kerner
// (dovrei andarmi a controllare i varlori di rirorno di ogni singolo thread...)



/// <summary>
/// Funzione per controllare che l'ultima esecuzione del kernel sia
/// andata a buon fine. Chiamare al termine della chiamata al kernel
/// (dopo previa sincronizzazione)
/// </summary>
/// <param name="msg">Messaggio libero per permettermi di capire a che riga sono</param>
static void checkKernelError(const char* msg) {

	// leggo errore ultima esecuzione kernel
	cudaError_t lastError = cudaGetLastError();

	if (lastError != cudaSuccess) {
		fprintf(stderr, "Errore nell'esecuzione dell'ultimo kernel cuda [%s]: %s\n%s",
			cudaGetErrorName(lastError), cudaGetErrorString(lastError), msg);

		exit(EXIT_FAILURE);
	}
}

//NOTA: occhio, non è detto che vengano colti i classici errori di accesso alla memoria

#endif

