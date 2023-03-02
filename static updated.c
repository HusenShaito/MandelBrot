#include <stdio.h>

#include <stdlib.h>

#include <math.h>

#include <mpi.h>

 

#define WIDTH 800

#define HEIGHT 600

#define MAX_ITER 1000

 

int mandelbrot(double x, double y) {

    double real = x;

    double imag = y;

    int i;

    for (i = 0; i < MAX_ITER; i++) {

        double r2 = real * real;

        double i2 = imag * imag;

        if (r2 + i2 > 4) {

            break;

        }

        imag = 2 * real * imag + y;

        real = r2 - i2 + x;

   }

    return i;

}

 

int main(int argc, char** argv) {

    double start, end;

    start = MPI_Wtime();

    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

 

    double xmin = -2.0;

    double xmax = 1.0;

    double ymin = -1.0;

    double ymax = 1.0;

 

    // Divide the image into 4 horizontal strips

    //static

    int num_per_process = HEIGHT / size; //600/4=150

    int low = rank * num_per_process; // 0->0, 1-> 150, 2->300, 3->450

    int high = (rank + 1) * num_per_process;// 0->150, 1->300, 2->450, 3->600

 

   

    // Allocate memory for the image strip

    int* image = (int*) malloc(WIDTH * (high - low) * sizeof(int));

 

    // Compute the Mandelbrot set for each part

    int x, y, iter;

    for (y = low; y < high; y++) {

        for (x = 0; x < WIDTH; x++) {

            double real = xmin + (double) x / WIDTH * (xmax - xmin);

            double imag = ymin + (double) y / HEIGHT * (ymax - ymin);

            iter = mandelbrot(real, imag);

            image[(y - low) * WIDTH + x] = iter;

        }

    }

   

    double Tcomm;

    // Gather the image strips and output the image

    if (rank == 0) {

        // Allocate memory for the final image

        int* final_image = (int*) malloc(WIDTH * HEIGHT * sizeof(int));

       

        double S = MPI_Wtime();

        MPI_Gather(image, WIDTH * num_per_process, MPI_INT, final_image, WIDTH * num_per_process, MPI_INT, 0, MPI_COMM_WORLD);

        double E = MPI_Wtime();

        Tcomm = E-S;

 

        // Output the image as a PPM file

        FILE* mandelbrot;

        mandelbrot = fopen("mandelbrot.pgm","wb");

        fprintf(mandelbrot, "P2\n");

        fprintf(mandelbrot, "%d %d\n", WIDTH, HEIGHT);

        fprintf(mandelbrot, "255\n");

        for (y = 0; y < HEIGHT; y++) {

            for (x = 0; x < WIDTH; x++) {

                iter = final_image[y * WIDTH + x];

                fprintf(mandelbrot, "%d ", iter);

            }

            fprintf(mandelbrot, "\n");

        }

        fclose(mandelbrot);

        free(final_image);

        end = MPI_Wtime();

printf("Overall Time = %f\n",end-start);

printf("Communication Time = %f\n",Tcomm);

    }

    else {

        double S = MPI_Wtime();

        MPI_Gather(image, WIDTH * num_per_process, MPI_INT, NULL, WIDTH * num_per_process, MPI_INT, 0, MPI_COMM_WORLD);

        double E = MPI_Wtime();

        Tcomm = Tcomm + E-S;

    }

 

free(image);

MPI_Finalize();

 

return 0;

}