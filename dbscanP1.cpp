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


// Version paralela P1 del algoritmo ingenuo de DBSCAN para detectar ruido
// Considera toda la matriz sin dividirla, cada hilo toma varios puntos i
void dbscanP1(double* x, double* y, int* etiqueta, int n, double epsilon, int minPuntos) {
    // Paso 1
    // Un punto es core de primer orden si tiene al menos minPuntos vecinos
    // a distancia epsilon contandose a si mismo, en otro caso es ruido
    // Cada punto i es independiente y solo escribe su propia etiqueta
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int vecinos = 0;
        // Cada punto mide su distancia contra todos los puntos
        for (int j = 0; j < n; j++) {
            if (sonVecinos(x, y, i, j, epsilon))
                vecinos++;
        }

        if (vecinos >= minPuntos)
            etiqueta[i] = COREPRIMERO;
        else
            etiqueta[i] = RUIDO;
    }

    // Copia del paso 1, para que en el paso 2 ningun hilo lea una etiqueta
    // que otro hilo esta cambiando
    int* etiquetaPaso1 = new int[n];
    for (int i = 0; i < n; i++)
        etiquetaPaso1[i] = etiqueta[i];

    // Paso 2
    // Un punto ruido se vuelve core de segundo orden si esta a distancia
    // epsilon de algun core de primer orden
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        if (etiquetaPaso1[i] == RUIDO) {
            int j = 0;
            // Se deja de buscar en cuanto encuentra un core de primer orden cercano
            while (j < n && etiqueta[i] == RUIDO) {
                if (etiquetaPaso1[j] == COREPRIMERO && sonVecinos(x, y, i, j, epsilon))
                    etiqueta[i] = CORESEGUNDO;
                j++;
            }
        }
    }

    delete[] etiquetaPaso1;
}


int main(int argc, char* argv[]) {
    // Parametros del experimento 
    int tamanos[8] = {20000, 40000, 80000, 120000, 140000, 160000, 180000, 200000};
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

    // Se puede dar un tamano y las repeticiones, por ejemplo dbscanP1 20000 1
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
    // Tiene el mismo formato que el de P2, con k igual a 1 porque la matriz no se divide
    ofstream archivoTiempos;
    if (corridaCompleta) {
        archivoTiempos.open("datos/tiempos_p1.csv");
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

            // Etiquetas que escribio el serial, para comprobar que P1 da lo mismo
            if (!leerEtiquetas(rutaSerial, etiquetaSerial, n)) {
                cout << "No se pudo leer " << rutaSerial << "\n";
                return 1;
            }

            // Se escribe el csv con los puntos core y ruido, sin contar en el tiempo
            // Todas las configuraciones dan las mismas etiquetas, por eso basta una
            if (corridaCompleta) {
                dbscanP1(x, y, etiqueta, n, epsilon, minPuntos);
                escribirResultados("datos/" + to_string(n) + "_results_p1.csv", x, y, etiqueta, n);
            }

            for (int h = 0; h < 4; h++) {
                omp_set_num_threads(hilos[h]);

                // Corrida de calentamiento, tambien sirve para comparar con el serial
                dbscanP1(x, y, etiqueta, n, epsilon, minPuntos);
                int diferencias = 0;
                for (int i = 0; i < n; i++)
                    if (etiqueta[i] != etiquetaSerial[i])
                        diferencias++;
                if (diferencias > 0)
                    cout << "MISMATCH puntos " << n << " hilos " << hilos[h]
                         << " diferencias " << diferencias << "\n";

                // Se corre el algoritmo varias veces y se promedia el tiempo
                double suma = 0;
                for (int r = 0; r < repeticiones; r++) {
                    double inicio = omp_get_wtime();
                    dbscanP1(x, y, etiqueta, n, epsilon, minPuntos);
                    double fin = omp_get_wtime();
                    suma = suma + (fin - inicio);

                    // Aviso cuando el experimento va a la mitad de sus repeticiones
                    if (r + 1 == repeticiones / 2)
                        cout << "puntos " << n << " hilos " << hilos[h] << " mitad "
                             << r + 1 << " de " << repeticiones << " repeticiones" << endl;
                }
                double promedio = suma / repeticiones;

                if (corridaCompleta)
                    archivoTiempos << n << "," << hilos[h] << ",1," << promedio << "\n";
                else
                    cout << "puntos " << n << " hilos " << hilos[h]
                         << " promedio " << promedio << " s\n";
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
