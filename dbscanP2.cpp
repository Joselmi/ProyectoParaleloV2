#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <cstdlib>
#include <omp.h>

using namespace std;

// Etiquetas que puede tener cada punto
const int RUIDO = 0;
const int COREPRIMERO = 1;
const int CORESEGUNDO = 2;

// Cuenta cuantos puntos hay en el archivo csv, uno por renglon
int contarPuntos(string ruta) {
    ifstream archivo(ruta);
    if (!archivo.is_open())
        return 0;

    int total = 0;
    double x, y;
    char coma;
    while (archivo >> x >> coma >> y)
        total++;

    archivo.close();
    return total;
}


// Lee las coordenadas x y de cada punto y las guarda en dos arreglos
void leerPuntos(string ruta, double* x, double* y, int n) {
    ifstream archivo(ruta);
    char coma;
    for (int i = 0; i < n; i++)
        archivo >> x[i] >> coma >> y[i];
    archivo.close();
}


// Lee la etiqueta de cada punto del csv que escribio el serial
bool leerEtiquetas(string ruta, int* etiqueta, int n) {
    ifstream archivo(ruta);
    if (!archivo.is_open())
        return false;

    double x, y;
    char coma1, coma2;
    for (int i = 0; i < n; i++)
        archivo >> x >> coma1 >> y >> coma2 >> etiqueta[i];
    archivo.close();
    return true;
}


// Escribe un csv con x, y y la etiqueta de cada punto
void escribirResultados(string ruta, double* x, double* y, int* etiqueta, int n) {
    ofstream archivo(ruta);
    archivo << fixed << setprecision(3);
    for (int i = 0; i < n; i++)
        archivo << x[i] << "," << y[i] << "," << etiqueta[i] << "\n";
    archivo.close();
}


// Regresa verdadero si los puntos i y j estan a distancia epsilon o menos
bool sonVecinos(double* x, double* y, int i, int j, double epsilon) {
    double distancia = sqrt((x[j] - x[i]) * (x[j] - x[i]) + (y[j] - y[i]) * (y[j] - y[i]));
    return distancia <= epsilon;
}


// Version paralela P2 del algoritmo ingenuo de DBSCAN para detectar ruido
// Divide el plano en k particiones y cada hilo trabaja una particion completa
// Cada particion guarda los indices de sus puntos y tambien los de las particiones de al lado
// que estan a distancia epsilon o menos de su borde (el halo), porque un vecino de un
// punto de la orilla puede estar del otro lado de la linea que separa dos particiones
void dbscanP2(double* x, double* y, int* etiqueta, int n, double epsilon, int minPuntos, int k) {
    // Se convierte k en una cuadricula de columnas por filas
    // k = 2 es 2x1, k = 4 es 2x2, k = 8 es 4x2, k = 16 es 4x4, k = 32 es 8x4 y k = 64 es 8x8
    int columnas = 1;
    int filas = 1;
    while (columnas * filas < k) {
        if (columnas <= filas)
            columnas = columnas * 2;
        else
            filas = filas * 2;
    }
    int numParticiones = columnas * filas;

    // Limites de los puntos para armar la cuadricula
    double minX = x[0];
    double maxX = x[0];
    double minY = y[0];
    double maxY = y[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < minX) minX = x[i];
        if (x[i] > maxX) maxX = x[i];
        if (y[i] < minY) minY = y[i];
        if (y[i] > maxY) maxY = y[i];
    }
    double anchoX = (maxX - minX) / columnas;
    double anchoY = (maxY - minY) / filas;

    // Particion a la que pertenece cada punto
    int* particion = new int[n];
    for (int i = 0; i < n; i++) {
        int cx = (int)((x[i] - minX) / anchoX);
        int cy = (int)((y[i] - minY) / anchoY);
        if (cx >= columnas) cx = columnas - 1;
        if (cy >= filas) cy = filas - 1;
        particion[i] = cy * columnas + cx;
    }

    // Lista de cada particion: indice de sus puntos y de los del halo
    int** indices = new int*[numParticiones];
    int* cuantos = new int[numParticiones];

    // Se pide memoria para n indices en cada particion, que es lo maximo que pueden ser
    for (int p = 0; p < numParticiones; p++)
        indices[p] = new int[n];

    // Armado de las listas, cada hilo arma la de una particion
    #pragma omp parallel for schedule(dynamic, 1)
    for (int p = 0; p < numParticiones; p++) {
        double izquierda = minX + (p % columnas) * anchoX - epsilon - 0.001;
        double derecha = minX + (p % columnas + 1) * anchoX + epsilon + 0.001;
        double abajo = minY + (p / columnas) * anchoY - epsilon - 0.001;
        double arriba = minY + (p / columnas + 1) * anchoY + epsilon + 0.001;

        // Se guardan los indices de los puntos de la particion y del halo
        cuantos[p] = 0;
        for (int i = 0; i < n; i++) {
            if (x[i] >= izquierda && x[i] <= derecha && y[i] >= abajo && y[i] <= arriba) {
                indices[p][cuantos[p]] = i;
                cuantos[p]++;
            }
        }
    }

    int* etiquetaPaso1 = new int[n];

    // Paso 1
    // Un punto es core de primer orden si tiene al menos minPuntos vecinos
    // a distancia epsilon contandose a si mismo, en otro caso es ruido
    // Cada hilo toma una particion y solo escribe la etiqueta de sus propios puntos
    #pragma omp parallel for schedule(dynamic, 1)
    for (int p = 0; p < numParticiones; p++) {
        for (int m = 0; m < cuantos[p]; m++) {
            int i = indices[p][m];
            // Los puntos del halo se clasifican en su propia particion
            if (particion[i] == p) {
                int vecinos = 0;
                // Cada punto mide su distancia contra todos los puntos de su particion y del halo
                for (int j = 0; j < cuantos[p]; j++) {
                    if (sonVecinos(x, y, i, indices[p][j], epsilon))
                        vecinos++;
                }

                if (vecinos >= minPuntos)
                    etiquetaPaso1[i] = COREPRIMERO;
                else
                    etiquetaPaso1[i] = RUIDO;
            }
        }
    }
    // El for paralelo espera a que todos los hilos terminen antes del paso 2,
    // porque un ruido puede tener cerca un core que es de otra particion

    // Paso 2
    // Un punto ruido se vuelve core de segundo orden si esta a distancia
    // epsilon de algun core de primer orden
    // Se lee de etiquetaPaso1, que ya no cambia, y se escribe en etiqueta
    #pragma omp parallel for schedule(dynamic, 1)
    for (int p = 0; p < numParticiones; p++) {
        for (int m = 0; m < cuantos[p]; m++) {
            int i = indices[p][m];
            if (particion[i] == p) {
                etiqueta[i] = etiquetaPaso1[i];
                if (etiquetaPaso1[i] == RUIDO) {
                    int j = 0;
                    // Se deja de buscar en cuanto encuentra un core de primer orden cercano
                    while (j < cuantos[p] && etiqueta[i] == RUIDO) {
                        int vecino = indices[p][j];
                        if (etiquetaPaso1[vecino] == COREPRIMERO && sonVecinos(x, y, i, vecino, epsilon))
                            etiqueta[i] = CORESEGUNDO;
                        j++;
                    }
                }
            }
        }
    }

    // Libera la memoria pedida
    for (int p = 0; p < numParticiones; p++)
        delete[] indices[p];
    delete[] indices;
    delete[] cuantos;
    delete[] particion;
    delete[] etiquetaPaso1;
}


int main(int argc, char* argv[]) {
    // Parametros del experimento 
    int tamanos[8] = {20000, 40000, 80000, 120000, 140000, 160000, 180000, 200000};
    int valoresK[6] = {2, 4, 8, 16, 32, 64};
    int repeticiones = 10;

    // Hilos: 1, cores virtuales entre 2, cores virtuales y cores virtuales por 2
    // Por defecto OpenMP crea un hilo por cada core virtual, el hilo 0 guarda cuantos son
    int cores = 0;
    int idHilo;
    #pragma omp parallel private(idHilo)
    {
        idHilo = omp_get_thread_num();
        if (idHilo == 0)
            cores = omp_get_num_threads();
    }
    int hilos[4] = {1, cores / 2, cores, cores * 2};

    // Mismos valores que usa el serial y la libreta
    double epsilon = 0.03;
    int minPuntos = 10;

    // Se puede dar un tamano y las repeticiones, por ejemplo dbscanP2 20000 1
    // Con un tamano solo se corre ese, se imprime en pantalla y no se escribe el archivo
    int tamanoUnico = 0;
    bool corridaCompleta = true;
    if (argc >= 2) {
        tamanoUnico = atoi(argv[1]);
        corridaCompleta = false;
    }
    if (argc >= 3)
        repeticiones = atoi(argv[2]);

    // Archivo con el tiempo promedio de cada configuracion
    ofstream archivoTiempos;
    if (corridaCompleta) {
        archivoTiempos.open("datos/tiempos_p2.csv");
        archivoTiempos << "puntos,hilos,k,promedio\n";
    }

    for (int t = 0; t < 8; t++) {
        // Con un tamano solo se corre ese
        if (corridaCompleta || tamanos[t] == tamanoUnico) {
            string rutaEntrada = "datos/" + to_string(tamanos[t]) + "_data.csv";
            string rutaSerial = "datos/" + to_string(tamanos[t]) + "_results.csv";

            int n = contarPuntos(rutaEntrada);
            if (n == 0) {
                cout << "No se pudo leer ningun punto de " << rutaEntrada << "\n";
                return 1;
            }

            // Memoria para las coordenadas y la etiqueta de cada punto
            double* x = new double[n];
            double* y = new double[n];
            int* etiqueta = new int[n];
            int* etiquetaSerial = new int[n];

            leerPuntos(rutaEntrada, x, y, n);

            // Etiquetas que escribio el serial, para comprobar que P2 da lo mismo
            if (!leerEtiquetas(rutaSerial, etiquetaSerial, n)) {
                cout << "No se pudo leer " << rutaSerial << "\n";
                return 1;
            }

            // Se escribe el csv con los puntos core y ruido, sin contar en el tiempo
            // Todas las configuraciones dan las mismas etiquetas, por eso basta una
            if (corridaCompleta) {
                dbscanP2(x, y, etiqueta, n, epsilon, minPuntos, 64);
                escribirResultados("datos/" + to_string(n) + "_results_p2.csv", x, y, etiqueta, n);
            }

            for (int a = 0; a < 6; a++) {
                int k = valoresK[a];
                for (int h = 0; h < 4; h++) {
                    omp_set_num_threads(hilos[h]);

                    // Corrida de calentamiento, tambien sirve para comparar con el serial
                    dbscanP2(x, y, etiqueta, n, epsilon, minPuntos, k);
                    int diferencias = 0;
                    for (int i = 0; i < n; i++)
                        if (etiqueta[i] != etiquetaSerial[i])
                            diferencias++;
                    if (diferencias > 0)
                        cout << "MISMATCH puntos " << n << " k " << k << " hilos " << hilos[h]
                             << " diferencias " << diferencias << "\n";

                    // Se corre el algoritmo varias veces y se promedia el tiempo
                    double suma = 0;
                    for (int r = 0; r < repeticiones; r++) {
                        double inicio = omp_get_wtime();
                        dbscanP2(x, y, etiqueta, n, epsilon, minPuntos, k);
                        double fin = omp_get_wtime();
                        suma = suma + (fin - inicio);

                        // Aviso cuando el experimento va a la mitad de sus repeticiones
                        if (r + 1 == repeticiones / 2)
                            cout << "puntos " << n << " k " << k << " hilos " << hilos[h] << " mitad "
                                 << r + 1 << " de " << repeticiones << " repeticiones" << endl;
                    }
                    double promedio = suma / repeticiones;

                    if (corridaCompleta)
                        archivoTiempos << n << "," << hilos[h] << "," << k << "," << promedio << "\n";
                    else
                        cout << "puntos " << n << " k " << k << " hilos " << hilos[h]
                             << " promedio " << promedio << " s\n";
                }
            }
            cout << n << " puntos listo\n";

            // Libera la memoria pedida
            delete[] x;
            delete[] y;
            delete[] etiqueta;
            delete[] etiquetaSerial;
        }
    }

    if (corridaCompleta)
        archivoTiempos.close();
    return 0;
}
