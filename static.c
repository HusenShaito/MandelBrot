#include <stdio.h>

#include <stdlib.h>

#include <math.h>

#include <mpi.h>
#include <time.h>


#define WIDTH 800

#define HEIGHT 600

#define MAX_ITER 1000

#define BAILOUT 2



int mandelbrot(double x, double y) {

    double real = x;

    double imag = y;

    int i;

    for (i = 0; i < MAX_ITER; i++) {

        double r2 = real * real;

        double i2 = imag * imag;

        if (r2 + i2 > BAILOUT * BAILOUT) {

            break;

        }

        imag = 2 * real * imag + y;

        real = r2 - i2 + x;

    }

    return i;

}



int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    clock_t husens,husene;
    double husenex;
    husens=clock();


    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);



    double xmin = -2.0;

    double xmax = 1.0;

    double ymin = -1.0;

    double ymax = 1.0;



    // Divide the image into 4 horizontal strips

    int strip_height = HEIGHT / size;

    int strip_start = rank * strip_height;

    int strip_end = (rank + 1) * strip_height;

    if (rank == size - 1) {

        strip_end = HEIGHT;

    }



    // Allocate memory for the image strip

    int* image = (int*) malloc(WIDTH * (strip_end - strip_start) * sizeof(int));



    // Compute the Mandelbrot set for the strip

    int x, y, iter;

    for (y = strip_start; y < strip_end; y++) {

        for (x = 0; x < WIDTH; x++) {

            double real = xmin + (double) x / WIDTH * (xmax - xmin);

            double imag = ymin + (double) y / HEIGHT * (ymax - ymin);

            iter = mandelbrot(real, imag);

            image[(y - strip_start) * WIDTH + x] = iter;

        }

    }



    // Gather the image strips and output the image

    if (rank == 0) {

        // Allocate memory for the final image

        int* final_image = (int*) malloc(WIDTH * HEIGHT * sizeof(int));

        MPI_Gather(image, WIDTH * strip_height, MPI_INT, final_image, WIDTH * strip_height, MPI_INT, 0, MPI_COMM_WORLD);



        // Output the image as a PPM file

        printf("P3\n%d %d\n255\n", WIDTH, HEIGHT);

        for (y = 0; y < HEIGHT; y++) {

            for (x = 0; x < WIDTH; x++) {

                iter = final_image[y * WIDTH + x];

                int r = (iter * 13) % 256;

                int g = (iter * 17) % 256;

                int b = (iter * 23) % 256;

                printf("%d %d %d    Running\n", r, g, b);

            }

        }



        free(final_image);

    } else {

        MPI_Gather(image, WIDTH * strip_height, MPI_INT, NULL, WIDTH * strip_height, MPI_INT, 0, MPI_COMM_WORLD);

}



free(image);
husene=clock();
husenex=((double) husene-husens)/CLOCKS_PER_SEC;
printf("\n\nThe Running time using %d proccessors is %f\n\n",size,husenex);
MPI_Finalize();



return 0;

}

