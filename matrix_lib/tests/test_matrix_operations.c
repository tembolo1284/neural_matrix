#include <criterion/criterion.h>
#include <criterion/logging.h>
#include <criterion/redirect.h>
#include <stdio.h>
#include "../include/matrix.h"  // Replace with your actual matrix library header

Test(matrix_operations, valid_index) {
    // Create a matrix for testing
    unsigned int num_rows = 3;
    unsigned int num_cols = 3;
    size_t element_size = sizeof(double);
    matrix *mat = matrix_new(num_rows, num_cols, element_size);
    
    // Fill the matrix with some values for testing
    double *data = (double *)mat->data;
    for (unsigned int i = 0; i < num_rows; i++) {
        for (unsigned int j = 0; j < num_cols; j++) {
            data[i * num_cols + j] = (double)(i * num_cols + j);
        }
    }
    
    // Test valid indices
    cr_assert_eq(matrix_at(mat, 0, 0), 0.0, "Value at (0, 0) is incorrect");
    cr_assert_eq(matrix_at(mat, 1, 1), 4.0, "Value at (1, 1) is incorrect");
    cr_assert_eq(matrix_at(mat, 2, 2), 8.0, "Value at (2, 2) is incorrect");

    matrix_free(mat); // Free the allocated matrix
}

Test(matrix_operations, eqdim_same_dimensions) {
    unsigned int rows = 3, cols = 4;
    matrix *mat1 = matrix_new(rows, cols, sizeof(double));
    matrix *mat2 = matrix_new(rows, cols, sizeof(double));

    cr_assert_not_null(mat1, "Matrix allocation for mat1 returned NULL");
    cr_assert_not_null(mat2, "Matrix allocation for mat2 returned NULL");

    // Ensure that mat1 and mat2 have the same dimensions
    cr_assert(matrix_eqdim(mat1, mat2), "matrix_eqdim failed for matrices with the same dimensions");

    matrix_free(mat1);
    matrix_free(mat2);
}

Test(matrix_operations, eqdim_different_dimensions) {
    unsigned int rows1 = 3, cols1 = 4;
    unsigned int rows2 = 4, cols2 = 3;
    matrix *mat1 = matrix_new(rows1, cols1, sizeof(double));
    matrix *mat2 = matrix_new(rows2, cols2, sizeof(double));

    cr_assert_not_null(mat1, "Matrix allocation for mat1 returned NULL");
    cr_assert_not_null(mat2, "Matrix allocation for mat2 returned NULL");

    // Ensure that mat1 and mat2 have different dimensions
    cr_assert_not(matrix_eqdim(mat1, mat2), "matrix_eqdim failed for matrices with different dimensions");

    matrix_free(mat1);
    matrix_free(mat2);
}

Test(matrix_operations, eq_within_tolerance) {
    unsigned int rows = 2, cols = 2;
    matrix *mat1 = matrix_new(rows, cols, sizeof(double));
    matrix *mat2 = matrix_new(rows, cols, sizeof(double));

    cr_assert_not_null(mat1, "Matrix allocation for mat1 returned NULL");
    cr_assert_not_null(mat2, "Matrix allocation for mat2 returned NULL");

    // Initialize mat1 and mat2 with values close to each other within tolerance
    double *data1 = (double *)mat1->data;
    double *data2 = (double *)mat2->data;
    data1[0] = 1.0;
    data1[1] = 2.0;
    data1[2] = 3.0;
    data1[3] = 4.0;

    data2[0] = 1.01; // Within tolerance of 0.01
    data2[1] = 2.01; // Within tolerance of 0.01
    data2[2] = 3.0;  // Exact match
    data2[3] = 4.0;  // Exact match

    double tolerance = 0.02; // Set your desired tolerance value

    // Ensure that mat1 and mat2 are equal within the specified tolerance
    cr_assert(matrix_eq(mat1, mat2, tolerance), "matrix_eq failed for matrices within tolerance");

    matrix_free(mat1);
    matrix_free(mat2);
}

Test(matrix_operations, eq_outside_tolerance) {
    unsigned int rows = 2, cols = 2;
    matrix *mat1 = matrix_new(rows, cols, sizeof(double));
    matrix *mat2 = matrix_new(rows, cols, sizeof(double));

    cr_assert_not_null(mat1, "Matrix allocation for mat1 returned NULL");
    cr_assert_not_null(mat2, "Matrix allocation for mat2 returned NULL");

    // Initialize mat1 and mat2 with values outside the specified tolerance
    double *data1 = (double *)mat1->data;
    double *data2 = (double *)mat2->data;
    data1[0] = 1.0;
    data1[1] = 2.0;
    data1[2] = 3.0;
    data1[3] = 4.0;

    data2[0] = 1.0;
    data2[1] = 2.0;
    data2[2] = 3.05;
    data2[3] = 4.0;

    double tolerance = 0.02; // Set your desired tolerance value

    // Ensure that mat1 and mat2 are not equal within the specified tolerance
    cr_assert_not(matrix_eq(mat1, mat2, tolerance), "matrix_eq failed for matrices outside tolerance");

    matrix_free(mat1);
    matrix_free(mat2);
}

Test(matrix_operations, is_symmetric_4x4) {
    unsigned int rows = 4, cols = 4;
    matrix *symmetric_mat = matrix_new(rows, cols, sizeof(double));

    // Initialize a symmetric matrix
    double symmetric_data[16] = {
        1.0, 2.0, 3.0, 4.0,
        2.0, 5.0, 6.0, 7.0,
        3.0, 6.0, 8.0, 9.0,
        4.0, 7.0, 9.0, 10.0
    };
    memcpy(symmetric_mat->data, symmetric_data, 16 * sizeof(double));

    cr_assert(matrix_is_symmetric(symmetric_mat), "Symmetric matrix check failed");

    matrix_free(symmetric_mat);
}

Test(matrix_operations, is_not_symmetric_4x4) {
    unsigned int rows = 4, cols = 4;
    matrix *non_symmetric_mat = matrix_new(rows, cols, sizeof(double));

    // Initialize a non-symmetric matrix
    double non_symmetric_data[16] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0
    };
    memcpy(non_symmetric_mat->data, non_symmetric_data, 16 * sizeof(double));

    cr_assert_not(matrix_is_symmetric(non_symmetric_mat), "Non-symmetric matrix check failed");

    matrix_free(non_symmetric_mat);
}

Test(matrix_operations, is_posdef_positive) {
    unsigned int rows = 3, cols = 3;
    matrix *posdef_mat = matrix_new(rows, cols, sizeof(double));

    // Initialize a positive definite matrix
    double posdef_data[9] = {
        4.0, 2.0, 1.0,
        2.0, 5.0, 2.0,
        1.0, 2.0, 6.0
    };
    memcpy(posdef_mat->data, posdef_data, 9 * sizeof(double));
   
    cr_assert(matrix_is_posdef(posdef_mat), "Positive definite matrix check failed");

    matrix_free(posdef_mat);
}


// Test for matrix_slice function for a single row
Test(matrix_operations, slice_single_row) {
    unsigned int rows = 3, cols = 4;
    matrix *mat = matrix_new(rows, cols, sizeof(double));

    // Initialize matrix data...

    Range single_row = {1, 2}; // 2nd row only
    Range all_cols = {-1, -1}; // All columns
    matrix *sliced_row = matrix_slice(mat, single_row, all_cols);

    cr_assert_not_null(sliced_row, "matrix_slice returned NULL for single row");
    cr_assert_eq(sliced_row->num_rows, 1, "Sliced row matrix should have 1 row");
    cr_assert_eq(sliced_row->num_cols, cols, "Sliced row matrix should have all columns");

    matrix_free(mat);
    matrix_free(sliced_row);
}

// Test for matrix_slice function for a single column
Test(matrix_operations, slice_single_column) {
    unsigned int rows = 3, cols = 4;
    matrix *mat = matrix_new(rows, cols, sizeof(double));

    // Initialize matrix data...

    Range all_rows = {-1, -1}; // All rows
    Range single_col = {2, 3}; // 3rd column only
    matrix *sliced_col = matrix_slice(mat, all_rows, single_col);

    cr_assert_not_null(sliced_col, "matrix_slice returned NULL for single column");
    cr_assert_eq(sliced_col->num_rows, rows, "Sliced column matrix should have all rows");
    cr_assert_eq(sliced_col->num_cols, 1, "Sliced column matrix should have 1 column");

    matrix_free(mat);
    matrix_free(sliced_col);
}

// Test for matrix_slice function for a range of rows and columns
Test(matrix_operations, slice_row_col_range) {
    unsigned int rows = 4, cols = 5;
    matrix *mat = matrix_new(rows, cols, sizeof(double));

    // Initialize matrix data...

    Range row_range = {1, 3}; // Rows 2 to 3
    Range col_range = {2, 4}; // Columns 3 to 4
    matrix *submatrix = matrix_slice(mat, row_range, col_range);

    cr_assert_not_null(submatrix, "matrix_slice returned NULL for row and column range");
    cr_assert_eq(submatrix->num_rows, 2, "Submatrix should have 2 rows");
    cr_assert_eq(submatrix->num_cols, 2, "Submatrix should have 2 columns");

    matrix_free(mat);
    matrix_free(submatrix);
}

Test(matrix_operations, slice_full_matrix) {
    unsigned int rows = 3, cols = 4;
    matrix *mat = matrix_new(rows, cols, sizeof(double));

    Range all_rows = {-1, -1};
    Range all_cols = {-1, -1};
    matrix *sliced = matrix_slice(mat, all_rows, all_cols);

    cr_assert_not_null(sliced, "matrix_slice returned NULL for full matrix");
    cr_assert_eq(sliced->num_rows, rows, "Sliced matrix should have all rows");
    cr_assert_eq(sliced->num_cols, cols, "Sliced matrix should have all columns");

    matrix_free(mat);
    matrix_free(sliced);
}


Test(matrix_operations, slice_all_rows_one_column) {
    unsigned int rows = 3, cols = 4, col_num = 1;
    matrix *mat = matrix_new(rows, cols, sizeof(double));

    Range all_rows = {-1, -1};
    Range single_col = {col_num, col_num + 1};
    matrix *sliced = matrix_slice(mat, all_rows, single_col);

    cr_assert_not_null(sliced, "matrix_slice returned NULL for all rows and one column");
    cr_assert_eq(sliced->num_rows, rows, "Sliced matrix should have all rows");
    cr_assert_eq(sliced->num_cols, 1, "Sliced matrix should have one column");

    matrix_free(mat);
    matrix_free(sliced);
}


Test(matrix_operations, slice_all_rows_range_of_columns) {
    unsigned int rows = 3, cols = 4;
    matrix *mat = matrix_new(rows, cols, sizeof(double));

    Range all_rows = {-1, -1};
    Range col_range = {1, 3};  // Columns 2 to 3
    matrix *sliced = matrix_slice(mat, all_rows, col_range);

    cr_assert_not_null(sliced, "matrix_slice returned NULL for all rows and range of columns");
    cr_assert_eq(sliced->num_rows, rows, "Sliced matrix should have all rows");
    cr_assert_eq(sliced->num_cols, 2, "Sliced matrix should have range of columns");

    matrix_free(mat);
    matrix_free(sliced);
}


Test(matrix_operations, slice_range_of_rows_one_column) {
    unsigned int rows = 4, cols = 5, col_num = 2;
    matrix *mat = matrix_new(rows, cols, sizeof(double));

    Range row_range = {1, 3};  // Rows 2 to 3
    Range single_col = {col_num, col_num + 1};
    matrix *sliced = matrix_slice(mat, row_range, single_col);
    cr_assert_not_null(sliced, "matrix_slice returned NULL for range of rows and one column");
    cr_assert_eq(sliced->num_rows, 2, "Sliced matrix should have range of rows");
    cr_assert_eq(sliced->num_cols, 1, "Sliced matrix should have one column");

    matrix_free(mat);
    matrix_free(sliced);
}

Test(matrix_operations, slice_range_of_rows_and_columns) {
    unsigned int rows = 4, cols = 5;
    matrix *mat = matrix_new(rows, cols, sizeof(double));

    Range row_range = {1, 3}; // Rows 2 to 3
    Range col_range = {2, 4}; // Columns 3 to 4
    matrix *sliced = matrix_slice(mat, row_range, col_range);

    cr_assert_not_null(sliced, "matrix_slice returned NULL for range of rows and columns");
    cr_assert_eq(sliced->num_rows, 2, "Sliced matrix should have range of rows");
    cr_assert_eq(sliced->num_cols, 2, "Sliced matrix should have range of columns");

    matrix_free(mat);
    matrix_free(sliced);
}

Test(matrix_operations, slice_one_row_one_column) {
    unsigned int rows = 3, cols = 4, row_num = 1, col_num = 2;
    matrix *mat = matrix_new(rows, cols, sizeof(double));

    Range single_row = {row_num, row_num + 1};
    Range single_col = {col_num, col_num + 1};
    matrix *sliced = matrix_slice(mat, single_row, single_col);

    cr_assert_not_null(sliced, "matrix_slice returned NULL for one row and one column");
    cr_assert_eq(sliced->num_rows, 1, "Sliced matrix should have one row");
    cr_assert_eq(sliced->num_cols, 1, "Sliced matrix should have one column");

    matrix_free(mat);
    matrix_free(sliced);
}

Test(matrix_operations, slice_one_row_range_of_columns) {
    unsigned int rows = 3, cols = 4, row_num = 1;
    matrix *mat = matrix_new(rows, cols, sizeof(double));

    Range single_row = {row_num, row_num + 1};
    Range col_range = {1, 3}; // Columns 2 to 3
    matrix *sliced = matrix_slice(mat, single_row, col_range);

    cr_assert_not_null(sliced, "matrix_slice returned NULL for one row and range of columns");
    cr_assert_eq(sliced->num_rows, 1, "Sliced matrix should have one row");
    cr_assert_eq(sliced->num_cols, 2, "Sliced matrix should have range of columns");

    matrix_free(mat);
    matrix_free(sliced);
}

Test(matrix_operations, swap_rows_square_matrix) {
    matrix *mat = matrix_new(3, 3, sizeof(double));
    cr_assert_not_null(mat, "Matrix allocation returned NULL");

    // Initialize matrix with some values
    double values[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    memcpy(mat->data, values, 9 * sizeof(double));

    matrix_swap_rows(mat, 0, 2); // Swap first and last rows

    // Verify that rows are swapped
    cr_assert_eq(((double*)mat->data)[0], 7, "First element of first row is incorrect after swap");
    cr_assert_eq(((double*)mat->data)[8], 3, "Last element of last row is incorrect after swap");

    matrix_free(mat);
}

// Test case for swapping two rows in a rectangular matrix
Test(matrix_operations, swap_rows_rectangular_matrix) {
    matrix *mat = matrix_new(4, 2, sizeof(double));
    cr_assert_not_null(mat, "Matrix allocation returned NULL");

    // Initialize matrix with some values
    double values[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    memcpy(mat->data, values, 8 * sizeof(double));

    matrix_swap_rows(mat, 1, 3); // Swap second and fourth rows

    // Verify that rows are swapped
    cr_assert_eq(((double*)mat->data)[2], 7, "First elem of second row is incorrect after swap");
    cr_assert_eq(((double*)mat->data)[7], 4, "Last elem of fourth row is incorrect after swap");

    matrix_free(mat);
}

// Test case for swapping two columns in a square matrix
Test(matrix_operations, swap_cols_square_matrix) {
    matrix *mat = matrix_new(3, 3, sizeof(double));
    cr_assert_not_null(mat, "Matrix allocation returned NULL");

    // Initialize matrix with some values
    double values[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    memcpy(mat->data, values, 9 * sizeof(double));

    matrix_swap_cols(mat, 0, 2); // Swap first and last columns

    // Verify that columns are swapped
    cr_assert_eq(((double*)mat->data)[0], 3, "First elem of first column is incorrect after swap");
    cr_assert_eq(((double*)mat->data)[8], 7, "Last elem of last column is incorrect after swap");

    matrix_free(mat);

}

// Test case for swapping two columns in a rectangular matrix
Test(matrix_operations, swap_cols_rectangular_matrix) {
    matrix *mat = matrix_new(2, 4, sizeof(double));
    cr_assert_not_null(mat, "Matrix allocation returned NULL");

    // Initialize matrix with some values
    double values[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    memcpy(mat->data, values, 8 * sizeof(double));

    matrix_swap_cols(mat, 1, 3); // Swap second and fourth columns

    // Verify that columns are swapped
    cr_assert_eq(((double*)mat->data)[1], 4, "Second elem of first row is incorrect after swap");
    cr_assert_eq(((double*)mat->data)[7], 6, "Last elem of second row is incorrect after swap");

    matrix_free(mat);

}

Test(matrix_operations, inverse_2x2_matrix) {
    // Create a 2x2 matrix
    matrix *mat = matrix_new(2, 2, sizeof(double));

    // Initialize the matrix with values for a simple test case
    double values[4] = {2.0, 1.0, 1.0, 3.0};
    memcpy(mat->data, values, 4 * sizeof(double));

    // Calculate the inverse of the matrix
    matrix *inverse = matrix_inv(mat);

    // Define the expected inverse matrix values
    matrix *inv_check = matrix_new(2, 2, sizeof(double));
    double expected_values[4] = {0.6, -0.2, -0.2, 0.4};
    memcpy(inv_check->data, expected_values, 4 * sizeof(double));

    // Verify that the inverse matrix is computed correctly
    cr_assert_not_null(inverse, "Matrix inversion returned NULL");
    cr_assert(matrix_eq(inverse, inv_check, 0.001), "Inverse matrix is incorrect");

    // Free the matrices
    matrix_free(mat);
    matrix_free(inverse);
    matrix_free(inv_check);
}

Test(matrix_operations, trace_2x2_matrix) {
    unsigned int rows = 2, cols = 2;
    matrix *mat = matrix_new(rows, cols, sizeof(double));

    // Initialize a 2x2 matrix with values
    double values[4] = {1.0, 2.0, 3.0, 4.0};
    memcpy(mat->data, values, 4 * sizeof(double));

    // Calculate the trace of the matrix
    double trace = matrix_trace(mat);

    // The trace of a 2x2 matrix is the sum of its diagonal elements
    double expected_trace = 1.0 + 4.0;

    // Assert that the calculated trace matches the expected value
    cr_assert_eq(trace, expected_trace, "Trace of the 2x2 matrix is incorrect");

    matrix_free(mat);
}

Test(matrix_operations, trace_3x3_matrix) {
    unsigned int rows = 3, cols = 3;
    matrix *mat = matrix_new(rows, cols, sizeof(double));

    // Initialize a 3x3 matrix with values
    double values[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    memcpy(mat->data, values, 9 * sizeof(double));

    // Calculate the trace of the matrix
    double trace = matrix_trace(mat);

    // The trace of a 3x3 matrix is the sum of its diagonal elements
    double expected_trace = 1.0 + 5.0 + 9.0;

    // Assert that the calculated trace matches the expected value
    cr_assert_eq(trace, expected_trace, "Trace of the 3x3 matrix is incorrect");

    matrix_free(mat);
}

Test(matrix_operations, cholesky_decomposition) {
    // Create a 3x3 symmetric positive-definite matrix
    unsigned int num_rows = 3;
    unsigned int num_cols = 3;
    size_t element_size = sizeof(double);
    matrix *mat = matrix_new(num_rows, num_cols, element_size);

    // Fill the matrix with values (must be symmetric and positive-definite)
    double *data = (double *)mat->data;
    data[0] = 9.0;
    data[1] = 3.0;
    data[2] = 6.0;
    data[3] = 3.0;
    data[4] = 5.0;
    data[5] = 4.0;
    data[6] = 6.0;
    data[7] = 4.0;
    data[8] = 9.0;

    // Perform Cholesky decomposition
    matrix_lup *cholesky = matrix_cholesky_solve(mat);

    // Check if Cholesky decomposition succeeded
    cr_assert_not_null(cholesky, "Cholesky decomposition failed for a valid matrix");

    // Check if the resulting matrix L is lower triangular
    matrix *L = cholesky->L;
    for (unsigned int i = 0; i < L->num_rows; i++) {
        for (unsigned int j = i + 1; j < L->num_cols; j++) {
            cr_assert_eq(matrix_at(L, i, j), 0.0, "Non-zero element in the upper triangle of L");
        }
    }

    // Clean up memory
    matrix_lup_free(cholesky);
    matrix_free(mat);
}

