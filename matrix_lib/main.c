#include "matrix.h"
#include <stdio.h>
#include <string.h>

int main() {
    // Allocate 2x3 double matrix
    matrix *mat = matrix_new(2, 3, sizeof(double));
    if (!mat) return 1;

    // Initialize values for double matrix
    double *data_double = (double *)mat->data;
    data_double[0] = 1.0;
    data_double[1] = 2.0;
    data_double[2] = 3.0;
    data_double[3] = 4.0;
    data_double[4] = 5.0;
    data_double[5] = 6.0;

    double mat_idx12 = matrix_at(mat, 0, 1);
    
    printf("printing value at location mat[1,2]: %lf\n", mat_idx12);
    // Print the double matrix
    printf("First matrix of doubles 2x3 from 1.0 to 6.0\n");
    matrix_print(mat);

    matrix_free(mat);
    
    matrix *mat1 = matrix_rand(3, 4, 0.0, 10.0, sizeof(double));
    printf("\nSecond matrix of random doubles 3x4 with values from 0.0-10.0\n");
    matrix_print(mat1);

    // Allocate 4x4 int square matrix
    matrix *mat2 = matrix_sqr(4, sizeof(double));
    if (!mat2) return 1;

    // Confirm we have an initial matrix of all zeroes
    printf("Check our 4x4 square matrix is initialized with all zeroes\n");
    matrix_print(mat2);

    // Initialize values for int matrix
    double *data_int = (double *)mat2->data;
    for (int i = 0; i < (int)(mat2->num_rows); i++) {
        for (int j = 0; j < (int)(mat2->num_cols); j++) {
            data_int[i * mat2->num_cols + j] = i + j + 1;
        }
    }

    // Print the int matrix
    printf("Confirm the square matrix is now from 1.0 to 16.0\n");
    matrix_print(mat2);

    // Free the matrix memory
    matrix_free(mat1);
    matrix_free(mat2);

    // Allocate 6x6 double identity matrix
    double identity_double = 1.0;
    matrix *mat3 = matrix_eye(6, sizeof(double), &identity_double);
    if (!mat3) return 1;

    printf("Confirm we have a proper 6x6 identity matrix\n");
    // Print the double identity matrix
    matrix_print(mat3);

    // Free the matrix memory
    matrix_free(mat3);

    Range all = { -1, -1 };
    Range single_row = { 1, 2 }; // 2nd row only
    Range single_col = { 2, 3 }; // 3rd column only
    Range row_range = { 1, 3 }; // Rows 2 to 3
    Range col_range = { 2, 4 }; // Columns 3 to 4

    
    matrix *mat0 = matrix_new(3, 5, sizeof(double));
    if (!mat) return 1;
  
    // Initialize values for double matrix
    double *data_double1 = (double *)mat0->data;
    data_double1[0] = 1.0;
    data_double1[1] = 2.0;
    data_double1[2] = 3.0;
    data_double1[3] = 4.0;
    data_double1[4] = 5.0;
    data_double1[5] = 6.0;
    data_double1[6] = 7.0;
    data_double1[7] = 8.0;
    data_double1[8] = 9.0;
    data_double1[9] = 10.0;
    data_double1[10] = 11.0;
    data_double1[11] = 12.0;
    data_double1[12] = 13.0;
    data_double1[13] = 14.0;
    data_double1[14] = 15.0;

    printf("Print 3x5 which should be from 1-15\n");
    matrix_print(mat0);

    // Get 2nd row, all columns
    printf("Print 2nd row, all columns\n");
    matrix *row_mat = matrix_slice(mat0, single_row, all);
    matrix_print(row_mat);

    // Get all rows, 3rd column
    printf("Print all rows, 3rd column\n");
    matrix *col_mat = matrix_slice(mat0, all, single_col);
    matrix_print(col_mat);

    // Get rows 2 to 3, all columns
    printf("Print rows 2 and 3, all columns\n");
    matrix *row_range_mat = matrix_slice(mat0, row_range, all);
    if (row_range_mat == NULL) {
        printf("Your matrix is NULL sir!\n");
    } else { 
            matrix_print(row_range_mat);
           }

    // Get all rows, columns 3 to 4
    printf("Print all rows, columns 3 and 4\n");
    matrix *col_range_mat = matrix_slice(mat0, all, col_range);
    matrix_print(col_range_mat);
    

    matrix_free(mat0);
    matrix_free(row_mat);
    matrix_free(col_mat);
    matrix_free(row_range_mat);
    matrix_free(col_range_mat);   

    matrix *mat_all = matrix_new(5, 4, sizeof(double));
    matrix *mat_diag = matrix_new(4, 4, sizeof(double)); 
    double val1 = 5.0;
    size_t value_size = sizeof(double);


    matrix_all_set(mat_all, &val1, value_size);
    matrix_diag_set(mat_diag, &val1, value_size);

    printf("Matrix should be all 5s\n");
    matrix_print(mat_all);

    printf("Matrix should have all diagonal entries of 5.0\n");
    matrix_print(mat_diag);

    matrix *mat_all_sliced = matrix_slice(mat_all, all, all);

    printf("Confirming matrix_slice all worked for matrix of all 5s.\n");
    matrix_print(mat_all_sliced);
  
    printf("Memory address of original matrix: %p\n", (void*)mat_all);
    printf("Memory address of copied matrix: %p\n", (void*)mat_all_sliced);

    
    matrix *mat_remove_row = matrix_row_rem(mat_all_sliced, 0);
    printf("Matrix_slice with first row removed. Should be 4x4 matrix of 5's now.\n");
    matrix_print(mat_remove_row);

    
    matrix *mat4 = matrix_new(3, 3, sizeof(double));
    double *data4 = (double *)mat4->data;
    for (unsigned int i = 0; i < 9; i++) {
        data4[i] = i + 1;
    }
    matrix *mat_remove_col = matrix_col_rem(mat4, 2);
    printf("Mat4 with last column removed. Should be 3x2 matrix now.\n");
    matrix_print(mat_remove_col);

    matrix *mat5 = matrix_new(3, 2, sizeof(double));
    double *data5 = (double *)mat5->data;
    for (unsigned int i = 0; i < 6; i++) {
        data5[i] = i + 2.25;
    }

    printf("Mat5 should be a 3x2 matrix of even numbers\n");
    matrix_print(mat5);

    matrix *mat6 = matrix_add(mat_remove_col, mat5);
    printf("Mat6 should be the sum of the two previous 3x2 matrices.\n");
    matrix_print(mat6);


    matrix *mat7 = matrix_subtract(mat6, mat5);
    printf("Mat7 should be last matrix minus matrix above last one.\n");
    matrix_print(mat7); 
  
    matrix *mat8 = matrix_row_rem(mat7, 2);
    matrix_print(mat8);

    
    matrix *ref = matrix_ref(mat8);

    printf("Mat8 in row echelon form is below:\n");
    matrix_print(ref);

    matrix_free(ref);    
    //matrix_free(mat7);
    matrix_free(mat5);
    matrix_free(mat6);
    matrix_free(mat8);

    
    matrix_free(mat_remove_row);
    matrix_free(mat_remove_col);
    matrix_free(mat_all);
    matrix_free(mat_diag);

    printf("LU decomposition time!\n");
    matrix *mat9 = matrix_new(2, 2, sizeof(double));

    ((double *)mat9->data)[0] = 2.0;
    ((double *)mat9->data)[1] = 1.0;
    ((double *)mat9->data)[2] = 1.0;
    ((double *)mat9->data)[3] = 3.0;

    printf("mat9 is below.\n");
    matrix_print(mat9);

    matrix_lup *lup = matrix_lup_solve(mat9);
    printf("Matrix L is:\n");
    matrix_print(lup->L);

    printf("Matrix U is:\n");
    matrix_print(lup->U);

    printf("Matrix P is:\n");
    matrix_print(lup->P);

    matrix_lup_free(lup);
    matrix_free(mat9);

    matrix *A = matrix_new(2, 2, sizeof(double));
    matrix *b = matrix_new(2, 1, sizeof(double));
    double *data_A = (double *)A->data;    
    double *data_b = (double *)b->data;

    for (unsigned int i = 0; i < 4; i++) {
        data_A[i] = i + 1.0;
    }

    data_b[0] = 2.0;
    data_b[1] = 3.0;

    printf("Matrix A is:\n");
    matrix_print(A);
    printf("Matrix b is:\n");
    matrix_print(b);

    matrix_lup *lu = matrix_lup_solve(A);
    matrix *x = matrix_ls_solve(lu, b);

    printf("Solution matrix x is:\n");
    matrix_print(x);

    matrix_free(A);
    matrix_free(b);
    matrix_free(x);
    matrix_lup_free(lu);

    return 0;
}



