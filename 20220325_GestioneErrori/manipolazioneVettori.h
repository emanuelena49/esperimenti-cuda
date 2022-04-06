#ifndef MANIP_VETTORI
#define MANIP_VETTORI

#include<stdio.h>


void stampaVettore(int* v, int vLength, char* vName) {

	for (size_t i = 0; i < vLength; i++)
	{
		printf("%s[%d]\t=\t%d\n", vName, i, v[i]);
	}
}

void copiaVettore(int* source, int* dest, int length) {

	for (size_t i = 0; i < length; i++)
	{
		dest[i] = source[i];
	}
}

#endif