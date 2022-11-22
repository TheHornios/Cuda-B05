/**
* ARQUITECTURA DE COMPUTADORES
* 2º Grado en Ingenieria Informatica
*
* Básico 5
*
* Alumno: Rodrigo Pascual Arnaiz 
* Fecha: 19/10/2020
*
*/
///////////////////////////////////////////////////////////////////////////
// includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
///////////////////////////////////////////////////////////////////////////
// defines
#define PI 3.141593f

///////////////////////////////////////////////////////////////////////////
// declaracion de funciones
// HOST: funcion llamada desde el host y ejecutada en el host

/**
* Funcion: propiedadesDispositivo
* Objetivo: Mustra las propiedades del dispositvo, esta funcion
* es ejecutada llamada y ejecutada desde el host
*
* Param: INT id_dispositivo -> ID del dispotivo
* Return: void
* propiedades del dispositivo CUDA
*/
__host__ void propiedadesDispositivo(int id_dispositivo)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, id_dispositivo);
	// calculo del numero de cores (SP)
	int cuda_cores = 0;
	int multi_processor_count = deviceProp.multiProcessorCount;
	int major = deviceProp.major;
	int minor = deviceProp.minor;
	char* arquitectura = (char*)"";
	switch (major)
	{
	case 1:
		//TESLA
		cuda_cores = 8;
		arquitectura = (char*)"TESLA";
		break;
	case 2:
		//FERMI
		arquitectura = (char*)"FERMI";
		if (minor == 0)
			cuda_cores = 32;
		else
			cuda_cores = 48;
		break;
	case 3:
		//KEPLER
		arquitectura = (char*)"KEPLER";
		cuda_cores = 192;
		break;
	case 5:
		//MAXWELL
		arquitectura = (char*)"MAXWELL";
		cuda_cores = 128;
		break;
	case 6:
		//PASCAL
		arquitectura = (char*)"PASCAL";
		cuda_cores = 64;
		break;
	case 7:
		//VOLTA
		arquitectura = (char*)"VOLTA";
		cuda_cores = 64;
		break;
	case 8:
		//AMPERE
		arquitectura = (char*)"AMPERE";
		cuda_cores = 128;
		break;
	default:
		arquitectura = (char*)"DESCONOCIDA";
		//DESCONOCIDA
		cuda_cores = 0;
		printf("!!!!!dispositivo desconocido!!!!!\n");
	}
	// presentacion de propiedades
	printf("***************************************************\n");
	printf("DEVICE %d: %s\n", id_dispositivo, deviceProp.name);
	printf("***************************************************\n");
	printf("> Capacidad de Computo \t\t\t: %d.%d\n", major, minor);
	printf("> Arquitectura CUDA \t\t\t: %s \n", arquitectura);
	printf("> No. de MultiProcesadores \t\t: %d \n",
		multi_processor_count);
	printf("> No. de CUDA Cores (%dx%d) \t\t: %d \n", cuda_cores,
		multi_processor_count, cuda_cores*
		multi_processor_count);
	printf("> No. max. de Hilos (por bloque) \t: %d \n",
		deviceProp.maxThreadsPerBlock);
	printf("***************************************************\n");

}

///////////////////////////////////////////////////////////////////////////
// KERNEL
/**
* Funcion: pi
* Objetivo: Funcion que calcula el numero pi utilizando reduccion paralela
*
* Param: INT  terminos->Cantidad de terminos 
* Param : INT * resultado->Puntero resultado 
* Param : INT * temporal->Array temporal para poder acceder a datos de otros hilos
* Return : void
*/
__global__ void pi(int terminos, float* resultado, float* temporal)
{
	// indice local de cada hilo -> kernel con un solo bloque de N hilos
	int my_id = threadIdx.x;
	// rellenamos el vector de datos aplicando cada uno de los terminos de la sucesion 
	temporal[my_id] =  ( 1 / pow( (my_id + 1.0), 2 ) );
	

	// sincronizamos para evitar riesgos de tipo RAW
	__syncthreads();

	// ******************
	// REDUCCION PARALELA
	// ******************
		int salto = terminos / 2;
		// realizamos log2(N) iteraciones
		while (salto > 0)
		{
			// en cada paso solo trabajan la mitad de los hilos
			if (my_id < salto)
			{
				temporal[my_id] = temporal[my_id] + temporal[my_id + salto];
			}
			// sincronizamos los hilos evitar riesgos de tipo RAW
			__syncthreads();
			salto /= 2;
		}

	// Solo el hilo no.'0' escribe el resultado final
	if (my_id == 0)
	{
		*resultado = sqrt( (temporal[0] * 6) );
	}


}
///////////////////////////////////////////////////////////////////////////
// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	
	// Declaración de variables
	int deviceCount;
	bool is_numero_valido = false;
	int numero_terminos;
	float hst_init = 0;
	float hst_result, * dev_result, * dev_temp;

	// Buscando dispositivos
	cudaGetDeviceCount(&deviceCount);

	
	// Mostrar propiedades por pantalla
	if (deviceCount == 0)
	{
		printf("!!!!!No se han encontrado dispositivos CUDA!!!!!\n");
		printf("<pulsa [INTRO] para finalizar>");
		getchar();
		return 1;
	} 
	else
	{
		printf("Se han encontrado <%d> dispositivos CUDA:\n", deviceCount);
		for (int id = 0; id < deviceCount; id++)
		{
			propiedadesDispositivo(id);
		}
	}

	
	// Pedir numero de terminos para sacar el numero pi
	do {
		printf("  Introduce el numero de terminos (potencia de 2):");
		is_numero_valido = scanf("%i", &numero_terminos);
		printf("\n");
		if (is_numero_valido ) 
		{
			is_numero_valido = numero_terminos % 2 == 0 && numero_terminos > 0;
			if( !is_numero_valido )
				printf("  ERROR -> Tiene que ser potencia de 2 y mayor que 0\n");
		}
		else 
		{
			printf("  ERROR -> Tiene que ser un numero\n");
		}
	} while (!is_numero_valido);

	// reserva de memoria en el device
	cudaMalloc((void**)&dev_result, sizeof(float));
	cudaMalloc((void**)&dev_temp, sizeof(float) * numero_terminos);

	// Copiar datos al dispositivo
	cudaMemcpy(dev_result, &hst_init, sizeof(float), cudaMemcpyHostToDevice);
	
	// Ejecutamos la funcion PI con la cantidad de hilos siendo el numero de terminos introducido por el usuario 
	pi <<<1, numero_terminos >> > (numero_terminos, dev_result, dev_temp);

	// Copiar datos del dispositivo al host
	cudaMemcpy(&hst_result, dev_result, sizeof(float), cudaMemcpyDeviceToHost);


	// Mostramos los datos 
	printf("> Valor de pi:\t\t%f%\n", PI);
	printf("> Valor calculado:\t%f%\n", hst_result);
	printf("> Error relativo:\t%f%%\n", (hst_result * 100 / PI) - 100);


	// Salida del programa
	time_t fecha;
	time(&fecha);
	printf("***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));
	printf("<pulsa [INTRO] para finalizar>");
	getchar();
	return 0;
}
///////////////////////////////////////////////////////////////////////////
