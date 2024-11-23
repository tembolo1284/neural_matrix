#include <criterion/criterion.h>
#include <criterion/logging.h>
#include "../include/matrix.h"
#include <stdio.h>

// Test case for multiplying a row by a scalar
Test(matrix_math, row_mult_r) {
    matrix *mat = matrix_new(3, 3, sizeof(double));
    cr_assert_not_null(mat, "Matrix allocation returned NULL");

    // Initialize matrix with some values
    double *data = (double *) mat->data;
    for (unsigned int i = 0; i < 9; i++) {
        data[i] = i + 1; // 1 to 9
    }

    double scalar = 2.0;
    matrix_row_mult_r(mat, 1, scalar); // Multiply second row by scalar

    // Verify that the second row is multiplied by scalar

    for (unsigned int i = 0; i < 3; i++) {
        cr_assert_eq(data[3 + i], (i + 4) * scalar, "Element at [1][%u] is not correctly multiplied", i);
    }

    matrix_free(mat);
}

// Test case for multiplying a column by a scalar
Test(matrix_math, col_mult_r) {
    matrix *mat = matrix_new(3, 3, sizeof(double));
    cr_assert_not_null(mat, "Matrix allocation returned NULL");

    // Initialize matrix with some values
    double *data = (double *) mat->data;
    for (unsigned int i = 0; i < 9; i++) {
        data[i] = i + 1; // 1 to 9
    }

    double scalar = 3.0;
    matrix_col_mult_r(mat, 2, scalar); // Multiply third column by scalar

    // Verify that the third column is multiplied by scalar
    for (unsigned int i = 0; i < 3; i++) {
        cr_assert_eq(data[i * 3 + 2], (i * 3 + 3) * scalar, "Element at [%u][2] is not correctly multiplied", i);
    }

    matrix_free(mat);
}

// Test case for multiplying the entire matrix by a scalar
Test(matrix_math, add_rows_square_matrix) {
    matrix *mat = matrix_new(3, 3, sizeof(double));
    double values[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    memcpy(mat->data, values, 9 * sizeof(double));

    matrix_row_addrow(mat, 0, 1, 2.0); // Add rows 0 and 1, store result in row 1

    // Check if the sum is correct in row 1
    cr_assert_eq(((double*)mat->data)[3], 6.0, "Row addition result is incorrect");
    cr_assert_eq(((double*)mat->data)[4], 9.0, "Row addition result is incorrect");
    cr_assert_eq(((double*)mat->data)[5], 12.0, "Row addition result is incorrect");

    matrix_free(mat);
}

Test(matrix_math, add_rows_rectangular_matrix) {
    matrix *mat = matrix_new(4, 2, sizeof(double));
    double values[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    memcpy(mat->data, values, 8 * sizeof(double));

    matrix_row_addrow(mat, 1, 2, 3.0); // Add rows 1 and 2, store result in row 2

    // Check if the sum is correct in row 1
    cr_assert_eq(((double*)mat->data)[4], 14.0, "Row addition result is incorrect");
    cr_assert_eq(((double*)mat->data)[5], 18.0, "Row addition result is incorrect");

    matrix_free(mat);
}

Test(matrix_math, add_rows_invalid_indices) {
    matrix *mat = matrix_new(2, 2, sizeof(double));
    double values[4] = {1.0, 2.0, 3.0, 4.0};
    memcpy(mat->data, values, 4 * sizeof(double));

    // Call with invalid indices and check if function handles it gracefully
    matrix_row_addrow(mat, 2, 3, 1.0); // Invalid indices

    // Check if the matrix is unchanged
    cr_assert_eq(((double*)mat->data)[0], 1.0, "Matrix should be unchanged");
    cr_assert_eq(((double*)mat->data)[1], 2.0, "Matrix should be unchanged");
    cr_assert_eq(((double*)mat->data)[2], 3.0, "Matrix should be unchanged");
    cr_assert_eq(((double*)mat->data)[3], 4.0, "Matrix should be unchanged");

    matrix_free(mat);
}

Test(matrix_math, add_two_square_matrices) {
    // Create two 3x3 matrices
    matrix *mat1 = matrix_new(3, 3, sizeof(double));
    matrix *mat2 = matrix_new(3, 3, sizeof(double));
    cr_assert_not_null(mat1, "Matrix 1 allocation returned NULL");
    cr_assert_not_null(mat2, "Matrix 2 allocation returned NULL");

    // Initialize matrices with values
    double values1[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double values2[9] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    memcpy(mat1->data, values1, 9 * sizeof(double));
    memcpy(mat2->data, values2, 9 * sizeof(double));

    // Perform matrix addition
    matrix *result = matrix_add(mat1, mat2);

    // Check if the result matrix is correct
    double *result_data = (double *) result->data;
    for (int i = 0; i < 9; i++) {
        cr_assert_eq(result_data[i], 10, "Element at index %d is not correct", i);
    }

    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(result);
}


Test(matrix_math, add_two_rectangular_matrices) {
    // Create two 2x4 matrices
    matrix *mat1 = matrix_new(2, 4, sizeof(double));
    matrix *mat2 = matrix_new(2, 4, sizeof(double));
    cr_assert_not_null(mat1, "Matrix 1 allocation returned NULL");
    cr_assert_not_null(mat2, "Matrix 2 allocation returned NULL");

    // Initialize matrices with values
    double values1[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    double values2[8] = {8, 7, 6, 5, 4, 3, 2, 1};
    memcpy(mat1->data, values1, 8 * sizeof(double));
    memcpy(mat2->data, values2, 8 * sizeof(double));

    // Perform matrix addition
    matrix *result = matrix_add(mat1, mat2);

    // Check if the result matrix is correct
    double *result_data = (double *) result->data;
    double expected_values[8] = {9, 9, 9, 9, 9, 9, 9, 9};
    for (int i = 0; i < 8; i++) {
        cr_assert_eq(result_data[i], expected_values[i], "Element at index %d is not correct", i);
    }

    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(result);
}


Test(matrix_math, subtract_two_square_matrices) {
    // Create two 3x3 matrices
    matrix *mat1 = matrix_new(3, 3, sizeof(double));
    matrix *mat2 = matrix_new(3, 3, sizeof(double));
    cr_assert_not_null(mat1, "Matrix 1 allocation returned NULL");
    cr_assert_not_null(mat2, "Matrix 2 allocation returned NULL");

    // Initialize matrices with values
    double values1[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    double values2[9] = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    memcpy(mat1->data, values1, 9 * sizeof(double));
    memcpy(mat2->data, values2, 9 * sizeof(double));

    // Perform matrix addition
    matrix *result = matrix_subtract(mat1, mat2);

    // Check if the result matrix is correct
    double *result_data = (double *) result->data;
    double expected_values[9] = {-8, -6, -4, -2, 0, 2, 4, 6, 8};
    for (int i = 0; i < 9; i++) {
        cr_assert_eq(result_data[i], expected_values[i], "Element at index %d is not correct", i);
    }

    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(result);
}


// Test case for matrix multiplication with square matrices
Test(matrix_math, matrix_mult_square_matrices) {
    matrix *mat1 = matrix_new(2, 2, sizeof(double));
    matrix *mat2 = matrix_new(2, 2, sizeof(double));

    double values1[4] = {1.0, 2.0, 3.0, 4.0};
    double values2[4] = {5.0, 6.0, 7.0, 8.0};

    memcpy(mat1->data, values1, 4 * sizeof(double));
    memcpy(mat2->data, values2, 4 * sizeof(double));

    matrix *result = matrix_mult(mat1, mat2);

    double *result_data = (double *)result->data;
    double expected_values[4] = {19.0, 22.0, 43.0, 50.0};

    for (int i = 0; i < 4; i++) {
        cr_assert_eq(result_data[i], expected_values[i], "Element at index %d is not correct", i);
    }

    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(result);
}

// Test case for matrix multiplication with rectangular matrices
Test(matrix_math, matrix_mult_rectangular_matrices) {
    matrix *mat1 = matrix_new(2, 3, sizeof(double));
    matrix *mat2 = matrix_new(3, 2, sizeof(double));

    double values1[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double values2[6] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

    memcpy(mat1->data, values1, 6 * sizeof(double));
    memcpy(mat2->data, values2, 6 * sizeof(double));

    matrix *result = matrix_mult(mat1, mat2);

    double *result_data = (double *)result->data;
    double expected_values[4] = {58.0, 64.0, 139.0, 154.0};

    for (int i = 0; i < 4; i++) {
        cr_assert_eq(result_data[i], expected_values[i], "Element at index %d is not correct", i);
    }

    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(result);
}

// Test case for pivotidx function
Test(matrix_math, pivotidx_test) {
    matrix *mat = matrix_new(3, 3, sizeof(double));
    double values[9] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    memcpy(mat->data, values, 9 * sizeof(double));

    unsigned int pivot = matrix_pivotidx(mat, 0, 0);

    cr_assert_eq(pivot, 2, "Pivot index is incorrect");

    matrix_free(mat);
}

// Test case for matrix_ref function
Test(matrix_math, matrix_ref_test) {
    matrix *mat = matrix_new(3, 3, sizeof(double));
    double values[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    memcpy(mat->data, values, 9 * sizeof(double));

    matrix *ref = matrix_ref(mat);

    // Check if the dimensions of the reference matrix are correct
    cr_assert_eq(ref->num_rows, 3, "Reference matrix has incorrect number of rows");
    cr_assert_eq(ref->num_cols, 3, "Reference matrix has incorrect number of columns");

    // Check if the values in the reference matrix match the expected values
    double *ref_data = (double *)ref->data;
    double expected_values[9] = {1.0, 1.142857, 1.285714, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < 9; i++) {
        cr_assert_float_eq(ref_data[i], expected_values[i], 1e-6, "Element at index %d in reference matrix is incorrect", i);
    }

    matrix_free(mat);
    matrix_free(ref);
}

// Test case for matrix_lup_new() function
Test(matrix_math, lup_new_test) {
    // Create a matrix
    matrix *mat = matrix_new(3, 3, sizeof(double));
    double values[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    memcpy(mat->data, values, 9 * sizeof(double));

    double val = 1.0;

    // Create L and U matrices (for simplicity, set L to the identity matrix and U to the original matrix)
    matrix *L = matrix_eye(3, sizeof(double), &val); // Identity matrix
    matrix *U = matrix_new(3, 3, sizeof(double));
    memcpy(U->data, values, 9 * sizeof(double)); // Same as the original matrix

    // Create a permutation matrix P (for simplicity, set P to the identity matrix)
    matrix *P = matrix_eye(3, sizeof(double), &val); // Identity matrix

    // Create an LUP decomposition
    matrix_lup *lup = matrix_lup_new(L, U, P, 0); // 0 permutations

    // Check if the LUP decomposition was successfully created
    cr_assert_not_null(lup, "LUP decomposition creation failed");

    // Check if the L, U, and P matrices are not NULL
    cr_assert_not_null(lup->L, "L matrix in LUP decomposition is NULL");
    cr_assert_not_null(lup->U, "U matrix in LUP decomposition is NULL");
    cr_assert_not_null(lup->P, "P matrix in LUP decomposition is NULL");

    // Check if the dimensions of L, U, and P matrices are correct
    cr_assert_eq(lup->L->num_rows, 3, "L matrix in LUP decomposition has incorrect number of rows");
    cr_assert_eq(lup->L->num_cols, 3, "L matrix in LUP decomposition has incorrect number of columns");
    cr_assert_eq(lup->U->num_rows, 3, "U matrix in LUP decomposition has incorrect number of rows");
    cr_assert_eq(lup->U->num_cols, 3, "U matrix in LUP decomposition has incorrect number of columns");
    cr_assert_eq(lup->P->num_rows, 3, "P matrix in LUP decomposition has incorrect number of rows");
    cr_assert_eq(lup->P->num_cols, 3, "P matrix in LUP decomposition has incorrect number of columns");

    // Check if the values in L, U, and P matrices are correct
    double *L_data = (double *)lup->L->data;
    double *U_data = (double *)lup->U->data;
    double *P_data = (double *)lup->P->data;

    // Expected values for L, U, and P matrices
    double expected_L[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}; // identity matrix
    double expected_U[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}; // same as U matrix
    double expected_P[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}; // identity matrix

    for (int i = 0; i < 9; i++) {
        cr_assert_float_eq(L_data[i], expected_L[i], 1e-6, "Element at index %d in L matrix is incorrect", i);
        cr_assert_float_eq(U_data[i], expected_U[i], 1e-6, "Element at index %d in U matrix is incorrect", i);
        cr_assert_float_eq(P_data[i], expected_P[i], 1e-6, "Element at index %d in P matrix is incorrect", i);
    }

    // Free the LUP decomposition and the original matrix
    matrix_lup_free(lup);
    matrix_free(mat);
}

// Test case for matrix_lup_solve() function
Test(matrix_math, lup_factorization_2x2_test) {
    // Create a 2x2 matrix
    matrix *mat = matrix_new(2, 2, sizeof(double));
    double values[4] = {2.0, 1.0, 1.0, 3.0};
    memcpy(mat->data, values, 4 * sizeof(double));

    // Create an LUP decomposition
    matrix_lup *lup = matrix_lup_solve(mat);

    // Check if the LUP decomposition was successfully created
    cr_assert_not_null(lup, "LUP decomposition creation failed");

    // Check if the L, U, and P matrices are not NULL
    cr_assert_not_null(lup->L, "L matrix in LUP decomposition is NULL");
    cr_assert_not_null(lup->U, "U matrix in LUP decomposition is NULL");
    cr_assert_not_null(lup->P, "P matrix in LUP decomposition is NULL");

    // Check if the dimensions of L, U, and P matrices are correct
    cr_assert_eq(lup->L->num_rows, 2, "L matrix in LUP decomposition has incorrect number of rows");
    cr_assert_eq(lup->L->num_cols, 2, "L matrix in LUP decomposition has incorrect number of columns");
    cr_assert_eq(lup->U->num_rows, 2, "U matrix in LUP decomposition has incorrect number of rows");
    cr_assert_eq(lup->U->num_cols, 2, "U matrix in LUP decomposition has incorrect number of columns");
    cr_assert_eq(lup->P->num_rows, 2, "P matrix in LUP decomposition has incorrect number of rows");
    cr_assert_eq(lup->P->num_cols, 2, "P matrix in LUP decomposition has incorrect number of columns");

    // Check if the values in L, U, and P matrices are correct
    double *L_data = (double *)lup->L->data;
    double *U_data = (double *)lup->U->data;
    double *P_data = (double *)lup->P->data;

    // Expected values for L, U, and P matrices
    double expected_L[4] = {1.0, 0.0, 0.5, 1.0};
    double expected_U[4] = {2.0, 1.0, 0.0, 2.5};
    double expected_P[4] = {1.0, 0.0, 0.0, 1.0};

    for (int i = 0; i < 4; i++) {
        cr_assert_float_eq(L_data[i], expected_L[i], 1e-6, "Element at index %d in L matrix is incorrect", i);
        cr_assert_float_eq(U_data[i], expected_U[i], 1e-6, "Element at index %d in U matrix is incorrect", i);
        cr_assert_float_eq(P_data[i], expected_P[i], 1e-6, "Element at index %d in P matrix is incorrect", i);
    }

    // Free the LUP decomposition and the original matrix
    matrix_lup_free(lup);
    matrix_free(mat);

}

Test(matrix_math, ls_solvefwd_2x2) {
    // Create a 2x2 lower triangular matrix L
    matrix *L = matrix_new(2, 2, sizeof(double));
    double L_values[4] = {1.0, 0.0, 2.0, 3.0};
    memcpy(L->data, L_values, 4 * sizeof(double));

    // Create a 2x1 column matrix b
    matrix *b = matrix_new(2, 1, sizeof(double));
    double b_values[2] = {4.0, 7.0};
    memcpy(b->data, b_values, 2 * sizeof(double));

    // Perform forward substitution
    matrix *x = matrix_ls_solvefwd(L, b);

    // Check if the solution vector is not NULL
    cr_assert_not_null(x, "Solution vector is NULL");

    // Check if the dimensions of the solution vector are correct
    cr_assert_eq(x->num_rows, 2, "Solution vector has incorrect number of rows");
    cr_assert_eq(x->num_cols, 1, "Solution vector has incorrect number of columns");

    // Check if the solution vector values are correct
    double *x_values = (double *)x->data;
    double expected_values[2] = {4.0, (-1.0/3.0)}; // Expected solution for this system

    for (int i = 0; i < 2; i++) {
        cr_assert_float_eq(x_values[i], expected_values[i], 1e-6, "Element at index %d in solution vector is incorrect", i);
    }

    // Free the matrices and the solution vector
    matrix_free(L);
    matrix_free(b);
    matrix_free(x);
}

Test(matrix_math, ls_solvefwd_3x3) {
    // Create a 3x3 lower triangular matrix L
    matrix *L = matrix_new(3, 3, sizeof(double));
    double L_values[9] = {1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0};
    memcpy(L->data, L_values, 9 * sizeof(double));

    // Create a 3x1 column matrix b
    matrix *b = matrix_new(3, 1, sizeof(double));
    double b_values[3] = {1.0, 8.0, 24.0};
    memcpy(b->data, b_values, 3 * sizeof(double));

    // Perform forward substitution
    matrix *x = matrix_ls_solvefwd(L, b);

    // Check if the solution vector is not NULL
    cr_assert_not_null(x, "Solution vector is NULL");

    // Check if the dimensions of the solution vector are correct
    cr_assert_eq(x->num_rows, 3, "Solution vector has incorrect number of rows");
    cr_assert_eq(x->num_cols, 1, "Solution vector has incorrect number of columns");

    // Check if the solution vector values are correct
    double *x_values = (double *)x->data;
    double expected_values[3] = {1.0, 2.0, (5.0/3.0)}; // Expected solution for this system

    for (int i = 0; i < 3; i++) {
        cr_assert_float_eq(x_values[i], expected_values[i], 1e-6, "Element at index %d in solution vector is incorrect", i);
    }

    // Free the matrices and the solution vector
    matrix_free(L);
    matrix_free(b);
    matrix_free(x);
}

Test(matrix_math, ls_solvebck_2x2) {
    // Create an upper triangular matrix U (2x2)
    matrix *U = matrix_new(2, 2, sizeof(double));
    double U_values[4] = {2.0, 3.0, 0.0, 5.0};
    memcpy(U->data, U_values, 4 * sizeof(double));

    // Create a column vector b (2x1)
    matrix *b = matrix_new(2, 1, sizeof(double));
    double b_values[2] = {10.0, 15.0};
    memcpy(b->data, b_values, 2 * sizeof(double));

    // Solve the linear system U * x = b using back substitution
    matrix *x = matrix_ls_solvebck(U, b);

    // Check if the solution vector is not NULL
    cr_assert_not_null(x, "Solution vector is NULL");

    // Check if the dimensions of the solution vector are correct
    cr_assert_eq(x->num_rows, 2, "Solution vector has incorrect number of rows");
    cr_assert_eq(x->num_cols, 1, "Solution vector has incorrect number of columns");

    // Check if the solution vector values are correct
    double *x_values = (double *)x->data;
    double expected_values[2] = {(1.0/2.0), 3.0};

    for (int i = 0; i < 2; i++) {
        cr_assert_float_eq(x_values[i], expected_values[i], 1e-6, "Element at index %d in solution vector is incorrect", i);
    }

    // Free the matrices and solution vector
    matrix_free(U);
    matrix_free(b);
    matrix_free(x);
}

Test(matrix_math, ls_solvebck_3x3) {
    // Create an upper triangular matrix U (3x3)
    matrix *U = matrix_new(3, 3, sizeof(double));
    double U_values[9] = {2.0, 3.0, 1.0, 0.0, 5.0, 4.0, 0.0, 0.0, 3.0};
    memcpy(U->data, U_values, 9 * sizeof(double));

    // Create a column vector b (3x1)
    matrix *b = matrix_new(3, 1, sizeof(double));
    double b_values[3] = {15.0, 24.0, 9.0};
    memcpy(b->data, b_values, 3 * sizeof(double));

    // Solve the linear system U * x = b using back substitution
    matrix *x = matrix_ls_solvebck(U, b);

    // Check if the solution vector is not NULL
    cr_assert_not_null(x, "Solution vector is NULL");

    // Check if the dimensions of the solution vector are correct
    cr_assert_eq(x->num_rows, 3, "Solution vector has incorrect number of rows");
    cr_assert_eq(x->num_cols, 1, "Solution vector has incorrect number of columns");

    // Check if the solution vector values are correct
    double *x_values = (double *)x->data;
    double expected_values[3] = {2.4, 2.4, 3.0};

    for (int i = 0; i < 3; i++) {
        cr_assert_float_eq(x_values[i], expected_values[i], 1e-6, "Element at index %d in solution vector is incorrect", i);
    }

    // Free the matrices and solution vector
    matrix_free(U);
    matrix_free(b);
    matrix_free(x);
}

Test(matrix_math, ls_solve_2x2) {
    // Create a 2x2 coefficient matrix A
    matrix *A = matrix_new(2, 2, sizeof(double));
    double A_values[4] = {2.0, 1.0, 1.0, 3.0};
    memcpy(A->data, A_values, 4 * sizeof(double));

    // Create a 2x1 column vector b
    matrix *b = matrix_new(2, 1, sizeof(double));
    double b_values[2] = {5.0, 7.0};
    memcpy(b->data, b_values, 2 * sizeof(double));

    matrix_lup *lu = matrix_lup_solve(A);
    // Solve the linear system Ax = b using matrix_ls_solve
    matrix *x = matrix_ls_solve(lu, b);

    // Check if the solution vector is not NULL
    cr_assert_not_null(x, "Solution vector is NULL");

    // Check if the dimensions of the solution vector are correct
    cr_assert_eq(x->num_rows, 2, "Solution vector has incorrect number of rows");
    cr_assert_eq(x->num_cols, 1, "Solution vector has incorrect number of columns");

    // Check if the solution vector values are correct
    double *x_values = (double *)x->data;
    double expected_values[2] = {1.6, 1.8}; // Expected solution for this system

    for (int i = 0; i < 2; i++) {
        cr_assert_float_eq(x_values[i], expected_values[i], 1e-6, "Element at index %d in solution vector is incorrect", i);
    }

    // Free the matrices and solution vector
    matrix_free(A);
    matrix_free(b);
    matrix_free(x);
}

Test(matrix_math, ls_solve_3x3) {
    // Create a 3x3 coefficient matrix A
    matrix *A = matrix_new(3, 3, sizeof(double));
    double A_values[9] = {2.0, 1.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 1.0};
    memcpy(A->data, A_values, 9 * sizeof(double));

    // Create a 3x1 column vector b
    matrix *b = matrix_new(3, 1, sizeof(double));
    double b_values[3] = {8.0, 7.0, 6.0};
    memcpy(b->data, b_values, 3 * sizeof(double));

    matrix_lup *lu = matrix_lup_solve(A);
    // Solve the linear system Ax = b using matrix_ls_solve
    matrix *x = matrix_ls_solve(lu, b);

    // Check if the solution vector is not NULL
    cr_assert_not_null(x, "Solution vector is NULL");

    // Check if the dimensions of the solution vector are correct
    cr_assert_eq(x->num_rows, 3, "Solution vector has incorrect number of rows");
    cr_assert_eq(x->num_cols, 1, "Solution vector has incorrect number of columns");

    // Check if the solution vector values are correct
    double *x_values = (double *)x->data;
    double expected_values[3] = {0.833333, 0.833333, 1.833333}; // Expected solution for this system

    for (int i = 0; i < 3; i++) {
        cr_assert_float_eq(x_values[i], expected_values[i], 1e-6, "Element at index %d in solution vector is incorrect", i);
    }

    // Free the matrices and solution vector
    matrix_free(A);
    matrix_free(b);
    matrix_free(x);
}


Test(matrix_operations, determinant_2x2_matrix) {
    matrix *mat = matrix_new(2, 2, sizeof(double));

    // Initialize the 2x2 matrix with specific values
    double *data = (double *)mat->data;
    data[0] = 2.0;
    data[1] = 3.0;
    data[2] = 1.0;
    data[3] = 4.0;

    // Calculate the determinant using the matrix_det function
    matrix_lup *lu = matrix_lup_solve(mat);
    double determinant = matrix_det(lu);

    // The determinant of the 2x2 matrix [2 3; 1 4] is 2*4 - 3*1 = 8 - 3 = 5
    cr_assert_float_eq(determinant, 5.0, 1e-6, "Determinant of 2x2 matrix is incorrect");

    // Free resources
    matrix_lup_free(lu);
    matrix_free(mat);
}

Test(matrix_operations, determinant_3x3_matrix) {
    matrix *mat = matrix_new(3, 3, sizeof(double));

    // Initialize the 3x3 matrix with specific values
    double *data = (double *)mat->data;
    data[0] = 2.0;
    data[1] = 0.0;
    data[2] = 1.0;
    data[3] = 3.0;
    data[4] = 2.0;
    data[5] = 5.0;
    data[6] = 1.0;
    data[7] = 4.0;
    data[8] = 3.0;

    // Calculate the determinant using the matrix_det function
    matrix_lup *lu = matrix_lup_solve(mat);
    double determinant = matrix_det(lu);

    // The determinant of the 3x3 matrix [2 0 1; 3 2 5; 1 4 3] can be calculated manually
    // as 2 * (2*3 - 5*4) - 0 + 1 * (3*4 - 2*1) = 6 - 0 + 10 = 16
    cr_assert_float_eq(determinant, -18.0, 1e-6, "Determinant of 3x3 matrix is incorrect");

    // Free resources
    matrix_lup_free(lu);
    matrix_free(mat);
}
