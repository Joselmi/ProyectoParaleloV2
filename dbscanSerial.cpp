#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <ctime>

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


// Regresa verdadero si los puntos i y j estan a distancia epsilon o menos
bool sonVecinos(double* x, double* y, int i, int j, double epsilon) {
    double distancia = sqrt((x[j] - x[i]) * (x[j] - x[i]) + (y[j] - y[i]) * (y[j] - y[i]));
    return distancia <= epsilon;
}


// Version serial del algoritmo ingenuo de DBSCAN para detectar ruido
void dbscanSerial(double* x, double* y, int* etiqueta, int n, double epsilon, int minPuntos) {
    // Paso 1
    // Un punto es core de primer orden si tiene al menos minPuntos vecinos
    // a distancia epsilon contandose a si mismo, en otro caso es ruido
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

    // Paso 2
    // Un punto ruido se vuelve core de segundo orden si esta a distancia
    // epsilon de algun core de primer orden
    for (int i = 0; i < n; i++) {
        if (etiqueta[i] == RUIDO) {
            int j = 0;
            // Se deja de buscar en cuanto encuentra un core de primer orden cercano
            while (j < n && etiqueta[i] == RUIDO) {
                if (etiqueta[j] == COREPRIMERO && sonVecinos(x, y, i, j, epsilon))
                    etiqueta[i] = CORESEGUNDO;
                j++;
            }
        }
    }
}


// Escribe un csv con x, y y la etiqueta de cada punto
void escribirResultados(string ruta, double* x, double* y, int* etiqueta, int n) {
    ofstream archivo(ruta);
    archivo << fixed << setprecision(3);
    for (int i = 0; i < n; i++)
        archivo << x[i] << "," << y[i] << "," << etiqueta[i] << "\n";
    archivo.close();
}


int main(int argc, char* argv[]) {
    // Letra de la computadora donde se corre, J o M, va al final de cada archivo
    if (argc < 2) {
        cout << "Falta la letra de la computadora, por ejemplo dbscanSerial J\n";
        return 1;
    }
    string equipo = argv[1];

    // Parametros del experimento tal como vienen en las instrucciones
    int tamanos[8] = {20000, 40000, 80000, 120000, 140000, 160000, 180000, 200000};
    int repeticiones = 10;

    // Mismos valores que usa la libreta
    double epsilon = 0.03;
    int minPuntos = 10;

    // Archivo con el tiempo promedio de cada tamano
    ofstream archivoTiempos("datos/tiempos_serial_" + equipo + ".csv");
    archivoTiempos << "puntos,promedio\n";

    for (int t = 0; t < 8; t++) {
        string rutaEntrada = "datos/" + to_string(tamanos[t]) + "_data.csv";
        string rutaSalida = "datos/" + to_string(tamanos[t]) + "_results_" + equipo + ".csv";

        int n = contarPuntos(rutaEntrada);
        if (n == 0) {
            cout << "No se pudo leer ningun punto de " << rutaEntrada << "\n";
            return 1;
        }

        // Memoria para las coordenadas y la etiqueta de cada punto
        double* x = new double[n];
        double* y = new double[n];
        int* etiqueta = new int[n];

        leerPuntos(rutaEntrada, x, y, n);

        // Corrida de calentamiento
        dbscanSerial(x, y, etiqueta, n, epsilon, minPuntos);

        // Se corre el algoritmo varias veces y se promedia el tiempo
        double suma = 0;
        for (int r = 0; r < repeticiones; r++) {
            clock_t inicio = clock();
            dbscanSerial(x, y, etiqueta, n, epsilon, minPuntos);
            clock_t fin = clock();
            suma = suma + double(fin - inicio) / CLOCKS_PER_SEC;

            // Aviso cuando el experimento va a la mitad de sus repeticiones
            if (r + 1 == repeticiones / 2)
                cout << "puntos " << n << " mitad " << r + 1 << " de " << repeticiones << " repeticiones" << endl;
        }
        double promedio = suma / repeticiones;
        archivoTiempos << n << "," << promedio << "\n";

        escribirResultados(rutaSalida, x, y, etiqueta, n);
        cout << n << " puntos listo\n";

        // Libera la memoria pedida
        delete[] x;
        delete[] y;
        delete[] etiqueta;
    }

    archivoTiempos.close();
    return 0;
}
