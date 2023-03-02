#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

#define WIDTH 1000
#define HEIGHT 1000
#define MAX_ITER 1000

int mandelbrot(double x0, double y0) {
    double x = 0, y = 0, xtemp;
    int iter = 0;
    while (x*x + y*y <= 2*2 && iter < MAX_ITER) {
        xtemp = x*x - y*y + x0;
        y = 2*x*y + y0;
        x = xtemp;
        iter++;
    }
    return iter;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
	clock_t husens,husene;
    double husenex;
    husens=clock();
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // calculate the range of rows to be computed by each process
    int rows_per_process = ceil((double)HEIGHT / size);
    int start_row = rank * rows_per_process;
    int end_row = fmin(start_row + rows_per_process, HEIGHT);

    // calculate the scale factors
    double xscale = 3.5 / WIDTH;
    double yscale = 2.0 / HEIGHT;

    // allocate memory for the RGB data and iteration count arrays
    int* iter_count = (int*)malloc((end_row - start_row) * WIDTH * sizeof(int));
    int* rgb = (int*)malloc((end_row - start_row) * WIDTH * sizeof(int));

    // calculate the Mandelbrot set for each pixel in the assigned rows
    int row, col, index;
    for (row = start_row; row < end_row; row++) {
        for (col = 0; col < WIDTH; col++) {
            double x0 = col * xscale - 2.5;
            double y0 = row * yscale - 1.0;
            iter_count[(row - start_row) * WIDTH + col] = mandelbrot(x0, y0);
        }
    }

    // gather the iteration count data from all processes into the root process (rank 0)
    int* all_iter_count = NULL;
    if (rank == 0) {
        all_iter_count = (int*)malloc(HEIGHT * WIDTH * sizeof(int));
    }
    MPI_Gather(iter_count, (end_row - start_row) * WIDTH, MPI_INT, all_iter_count, (end_row - start_row) * WIDTH, MPI_INT, 0, MPI_COMM_WORLD);

    // calculate the RGB data for each pixel using the iteration count
    if (rank == 0) {
        int i, j;
        for (i = 0; i < HEIGHT; i++) {
            for (j = 0; j < WIDTH; j++) {
                index = i * WIDTH + j;
                int iter = all_iter_count[index];
                if (iter == MAX_ITER) {
                    rgb[3*index] = 0;
                    rgb[3*index+1] = 0;
                    rgb[3*index+2] = 0;
                } else {
                    double ratio = (double)iter / MAX_ITER;
                    rgb[3*index] = (int)(255 * pow(ratio, 3));
                    rgb[3*index+1] = (int)(255 * pow(ratio, 2));
                    rgb[3*index+2] = (int)(255 * ratio);
                }
            }
        }

       
// print the RGB data for the first pixel in each row (for testing)
for (i = 0; i < HEIGHT; i++) {
    index = i * WIDTH;
    printf("(%d,%d,%d) Running\n", rgb[3*index], rgb[3*index+1], rgb[3*index+2]);
}
printf("\n");

}
husene=clock();
husenex=((double) husene-husens)/CLOCKS_PER_SEC;
printf("\n\nThe Running time using %d proccessors is %f\n\n",size,husenex);
MPI_Finalize();
return 0;

}