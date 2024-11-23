#include <criterion/criterion.h>
#include <criterion/logging.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include "../include/matrix.h"  

// Test case for matrix allocation
Test(matrix_init, new_3by4) {
    int rows = 3, cols = 4;
    matrix *mat = matrix_new(rows, cols, sizeof(double));

    cr_assert_not_null(mat, "Matrix allocation returned NULL");
    cr_assert_eq(mat->num_rows, rows, "Number of rows is not correct");
    cr_assert_eq(mat->num_cols, cols, "Number of cols is not correct");

    // Further assertions can be added here as needed

    matrix_free(mat);  // Assuming this is your deallocation function
}

Test(matrix_operations, set_2x2_matrix_element) {
    // Create a 2x2 matrix for testing
    unsigned int num_rows = 2;
    unsigned int num_cols = 2;
    size_t element_size = sizeof(double);
    matrix *mat = matrix_new(num_rows, num_cols, element_size);

    // Set an element at a specific position (e.g., row 1, column 0) to a value
    double value = 42.425;
    unsigned int row_index = 1;
    unsigned int col_index = 0;
    matrix_set(mat, row_index, col_index, value);

    // Check if the element at the specified position is equal to the set value
    cr_assert_eq(matrix_at(mat, row_index, col_index), value);

    matrix_free(mat); // Free the allocated matrix
}

Test(matrix_operations, set_3x3_matrix_element) {
    // Create a 3x3 matrix for testing
    unsigned int num_rows = 3;
    unsigned int num_cols = 3;
    size_t element_size = sizeof(double);
    matrix *mat = matrix_new(num_rows, num_cols, element_size);

    // Set an element at a specific position (e.g., row 2, column 1) to a value
    double value = 88.88;
    unsigned int row_index = 2;
    unsigned int col_index = 1;
    matrix_set(mat, row_index, col_index, value);

    // Check if the element at the specified position is equal to the set value
    cr_assert_eq(matrix_at(mat, row_index, col_index), value);

    matrix_free(mat); // Free the allocated matrix
}


// Test case 1: Set all elements of a matrix to 0
Test(matrix_init, all_set_to_zero) {
    // Create a matrix for testing
    unsigned int num_rows = 3;
    unsigned int num_cols = 3;
    size_t element_size = sizeof(double);
    matrix *mat = matrix_new(num_rows, num_cols, element_size);

    // Set all elements to 0
    double zero = 0.0;
    matrix_all_set(mat, &zero, sizeof(double));

    // Check if all elements are 0
    for (unsigned int i = 0; i < num_rows; ++i) {
        for (unsigned int j = 0; j < num_cols; ++j) {
            cr_assert_eq(matrix_at(mat, i, j), 0.0);
        }
    }

    matrix_free(mat); // Free the allocated matrix
}

// Test case 2: Set all elements of a matrix to a specific value
Test(matrix_init, all_set_to_value) {
    // Create a matrix for testing
    unsigned int num_rows = 2;
    unsigned int num_cols = 2;
    size_t element_size = sizeof(double);
    matrix *mat = matrix_new(num_rows, num_cols, element_size);

    // Set all elements to a specific value (e.g., 42)
    double value = 42.0;
    matrix_all_set(mat, &value, sizeof(double));
    // Check if all elements are equal to the specified value
    for (unsigned int i = 0; i < num_rows; ++i) {
        for (unsigned int j = 0; j < num_cols; ++j) {
            cr_assert_eq(matrix_at(mat, i, j), 42.0);
        }
    }

    matrix_free(mat); // Free the allocated matrix
}

Test(matrix_operations, diag_set) {
    // Create a matrix for testing
    unsigned int num_rows = 3;
    unsigned int num_cols = 3;
    size_t element_size = sizeof(double);
    matrix *mat = matrix_new(num_rows, num_cols, element_size);

    // Set the diagonal elements to a specific value (e.g., 7.0)
    double value = 7.0;
    matrix_diag_set(mat, &value, sizeof(double));

    // Check if the diagonal elements are equal to the specified value
    for (unsigned int i = 0; i < num_rows; ++i) {
        for (unsigned int j = 0; j < num_cols; ++j) {
            if (i == j) {
                cr_assert_eq(matrix_at(mat, i, j), 7.0);
            } else {
                // Check that non-diagonal elements are not modified
                cr_assert_neq(matrix_at(mat, i, j), 7.0);
            }
        }
    }

    matrix_free(mat); // Free the allocated matrix
}



Test(matrix_init, random_values_within_range) {
    unsigned int num_rows = 3;
    unsigned int num_cols = 3;
    double min = 0.0;
    double max = 1.0;
    size_t element_size = sizeof(double);

    matrix *result = matrix_rand(num_rows, num_cols, min, max, element_size);

    // Check if the matrix is not NULL
    cr_assert_not_null(result, "matrix_rand() returned NULL");

    // Check if the matrix dimensions are as expected
    cr_assert_eq(result->num_rows, num_rows, "Unexpected number of rows");
    cr_assert_eq(result->num_cols, num_cols, "Unexpected number of columns");

    // Check if all elements are within the specified range
    double *data = (double *)result->data;
    for (int i = 0; i < (int)num_rows; i++) {
        for (int j = 0; j < (int)num_cols; j++) {
            cr_assert_geq(data[i * num_cols + j], min, "Value below the minimum range");
            cr_assert_leq(data[i * num_cols + j], max, "Value above the maximum range");
        }
    }

    matrix_free(result); // Free the allocated matrix
}

Test(matrix_init, invalid_allocation) {
    // Test 1: Moderately large allocation that should fail
    unsigned int num_rows = 50000;
    unsigned int num_cols = 50000;
    double min = 0.0;
    double max = 1.0;
    size_t element_size = sizeof(double);

    // This should fail as it would require ~20GB of memory
    matrix *result = matrix_rand(num_rows, num_cols, min, max, element_size);
    cr_assert_null(result, "matrix_rand() should return NULL for large allocation (20GB)");

    // Test 2: Even larger allocation
    num_rows = 100000;
    num_cols = 100000;
    result = matrix_rand(num_rows, num_cols, min, max, element_size);
    cr_assert_null(result, "matrix_rand() should return NULL for very large allocation (80GB)");

    // Test 3: Allocation that would cause integer overflow
    num_rows = UINT_MAX;
    num_cols = 2;
    result = matrix_rand(num_rows, num_cols, min, max, element_size);
    cr_assert_null(result, "matrix_rand() should return NULL for allocation that would overflow");
}

Test(matrix_init, square_matrix) {
    unsigned int size = 4;
    matrix *mat = matrix_sqr(size, sizeof(double));

    cr_assert_not_null(mat, "matrix_sqr returned NULL");
    cr_assert_eq(mat->num_rows, size, "Square matrix does not have the correct number of rows");
    cr_assert_eq(mat->num_cols, size, "Square matrix does not have the correct number of columns");
    cr_assert(mat->is_square, "Matrix is not square when it should be");

    matrix_free(mat);
}

Test(matrix_init, identity_matrix) {
    unsigned int size = 4;
    double identity_element = 1.0;
    matrix *mat = matrix_eye(size, sizeof(double), &identity_element);

    cr_assert_not_null(mat, "matrix_eye returned NULL");
    cr_assert_eq(mat->num_rows, size, "Identity matrix does not have the correct number of rows");
    cr_assert_eq(mat->num_cols, size, "Identity matrix does not have the correct number of columns");
    cr_assert(mat->is_square, "Matrix is not square when it should be");

    // Check if the diagonal elements are 1 and others are 0
    double *data = (double *)mat->data;
    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < size; j++) {
            double expected_value = (i == j) ? identity_element : 0.0;
            cr_assert_eq(data[i * size + j], expected_value, "Element at [%u][%u] is not correct", i, j);
        }
    }

    matrix_free(mat);
}

Test(matrix_init, print) {
    matrix *mat = matrix_new(2, 2, sizeof(double));
    if (!mat) {
      cr_assert_fail("Matrix allocation failed");
    }

    // Cast the void* data to double* for easy manipulation
    double *data = (double *) mat->data;

    // Initialize matrix with some values
    data[0] = 1.0;  // element at row 0, col 0
    data[1] = 2.0;  // element at row 0, col 1
    data[2] = 3.0;  // element at row 1, col 0
    data[3] = 4.0;  // element at row 1, col 1

    // Printing the matrix
    cr_log_info("Printing matrix:\n");

    matrix_free(mat);
}

Test(matrix_init, transpose_square_matrix) {
    matrix *mat = matrix_new(2, 2, sizeof(double));
    double values[4] = {1.0, 2.0, 3.0, 4.0};
    memcpy(mat->data, values, 4 * sizeof(double));

    matrix_transpose(mat);

    cr_assert_eq(((double*)mat->data)[0], 1.0, "Incorrect value in transposed matrix");
    cr_assert_eq(((double*)mat->data)[1], 3.0, "Incorrect value in transposed matrix");
    cr_assert_eq(((double*)mat->data)[2], 2.0, "Incorrect value in transposed matrix");
    cr_assert_eq(((double*)mat->data)[3], 4.0, "Incorrect value in transposed matrix");

    matrix_free(mat);
}

Test(matrix_init, transpose_rectangular_matrix) {
    matrix *mat = matrix_new(2, 4, sizeof(double));
    double values[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    memcpy(mat->data, values, 8 * sizeof(double));

    matrix_transpose(mat);

    cr_assert_eq(mat->num_rows, 4, "Number of rows should be 3 after transpose");
    cr_assert_eq(mat->num_cols, 2, "Number of columns should be 2 after transpose");
    cr_assert_eq(((double*)mat->data)[0], 1.0, "Incorrect value in transposed matrix");
    cr_assert_eq(((double*)mat->data)[1], 5.0, "Incorrect value in transposed matrix");
    cr_assert_eq(((double*)mat->data)[2], 2.0, "Incorrect value in transposed matrix");
    cr_assert_eq(((double*)mat->data)[3], 6.0, "Incorrect value in transposed matrix");
    cr_assert_eq(((double*)mat->data)[4], 3.0, "Incorrect value in transposed matrix");
    cr_assert_eq(((double*)mat->data)[5], 7.0, "Incorrect value in transposed matrix");
    cr_assert_eq(((double*)mat->data)[6], 4.0, "Incorrect value in transposed matrix");
    cr_assert_eq(((double*)mat->data)[7], 8.0, "Incorrect value in transposed matrix");

    matrix_free(mat);
}

Test(matrix_init, stackv_square_matrices) {
    matrix *mat1 = matrix_new(2, 2, sizeof(double));
    matrix *mat2 = matrix_new(2, 2, sizeof(double));
    double values1[4] = {1.0, 2.0, 3.0, 4.0};
    double values2[4] = {5.0, 6.0, 7.0, 8.0};
    memcpy(mat1->data, values1, 4 * sizeof(double));
    memcpy(mat2->data, values2, 4 * sizeof(double));

    matrix *result = matrix_stackv(mat1, mat2);

    cr_assert_not_null(result, "matrix_stackv returned NULL");
    cr_assert_eq(result->num_rows, 4, "Stacked matrix should have 4 rows");
    cr_assert_eq(result->num_cols, 2, "Stacked matrix should have 2 columns");

    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(result);
}


Test(matrix_init, stackv_rectangular_matrices) {
    matrix *mat1 = matrix_new(2, 3, sizeof(double));
    matrix *mat2 = matrix_new(2, 3, sizeof(double));
    // Initialize matrices mat1 and mat2 with some values

    matrix *result = matrix_stackv(mat1, mat2);

    cr_assert_not_null(result, "matrix_stackv returned NULL");
    cr_assert_eq(result->num_rows, 4, "Stacked matrix should have 4 rows");
    cr_assert_eq(result->num_cols, 3, "Stacked matrix should have 3 columns");

    // Optionally, add assertions to check the values in the stacked matrix
   
    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(result);

}


Test(matrix_init, stackh_square_matrices) {
    matrix *mat1 = matrix_new(2, 2, sizeof(double));
    matrix *mat2 = matrix_new(2, 2, sizeof(double));
    double values1[4] = {1.0, 2.0, 3.0, 4.0};
    double values2[4] = {5.0, 6.0, 7.0, 8.0};
    memcpy(mat1->data, values1, 4 * sizeof(double));
    memcpy(mat2->data, values2, 4 * sizeof(double));

    matrix *result = matrix_stackh(mat1, mat2);

    cr_assert_not_null(result, "matrix_stackh returned NULL");
    cr_assert_eq(result->num_rows, 2, "Stacked matrix should have 2 rows");
    cr_assert_eq(result->num_cols, 4, "Stacked matrix should have 4 columns");

    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(result);
}

Test(matrix_init, stackh_rectangular_matrices) {
    matrix *mat1 = matrix_new(3, 2, sizeof(double));
    matrix *mat2 = matrix_new(3, 2, sizeof(double));
    // Initialize matrices mat1 and mat2 with some values

    matrix *result = matrix_stackh(mat1, mat2);

    cr_assert_not_null(result, "matrix_stackh returned NULL");
    cr_assert_eq(result->num_rows, 3, "Stacked matrix should have 3 rows");
    cr_assert_eq(result->num_cols, 4, "Stacked matrix should have 4 columns");

    // Optionally, add assertions to check the values in the stacked matrix
    
    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(result);
}


// Test case for removing a row
Test(matrix_init, row_remove_first) {
    matrix *mat = matrix_new(3, 3, sizeof(double));
    cr_assert_not_null(mat, "Matrix allocation returned NULL");

    // Initialize matrix with some values
    double *data = (double *)mat->data;
    for (unsigned int i = 0; i < 9; i++) {
        data[i] = i + 1; // 1 to 9
    }

    // Remove the first row
    matrix *new_mat = matrix_row_rem(mat, 0);

    // Verify the new matrix dimensions and contents
    cr_assert_not_null(new_mat, "New matrix after row removal is NULL");
    cr_assert_eq(new_mat->num_rows, 2, "New matrix should have 2 rows");
    cr_assert_eq(new_mat->num_cols, 3, "Number of columns should remain unchanged");
    double expected_values[6] = {4, 5, 6, 7, 8, 9};
    for (unsigned int i = 0; i < 6; i++) {
        cr_assert_eq(((double *)new_mat->data)[i], expected_values[i], "Element [%u] is incorrect after row removal", i);
    }

    matrix_free(new_mat);
}

Test(matrix_init, row_remove_last) {

    matrix *mat = matrix_new(3, 4, sizeof(double));
    cr_assert_not_null(mat, "Matrix allocation returned NULL");

    // Initialize matrix with some values
    double *data = (double *)mat->data;
    for (unsigned int i = 0; i < 12; i++) {
        data[i] = i + 1; // 1 to 12
    }

    // Remove the last row
    matrix *new_mat = matrix_row_rem(mat, 2);

    // Verify the new matrix dimensions and contents
    cr_assert_not_null(new_mat, "New matrix after row removal is NULL");
    cr_assert_eq(new_mat->num_rows, 2, "New matrix should have 2 rows");
    cr_assert_eq(new_mat->num_cols, 4, "Number of columns should remain unchanged");
    double expected_values[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    for (unsigned int i = 0; i < 8; i++) {
        cr_assert_eq(((double *)new_mat->data)[i], expected_values[i], "Element [%u] is incorrect after row removal", i);
    }

    matrix_free(new_mat);
}

Test(matrix_init, col_remove_first) {

    matrix *mat = matrix_new(3, 4, sizeof(double));
    cr_assert_not_null(mat, "Matrix allocation returned NULL");

    // Initialize matrix with some values
    double *data = (double *)mat->data;
    for (unsigned int i = 0; i < 12; i++) {
        data[i] = i + 1; // 1 to 12
    }

    // Remove the last row
    matrix *new_mat = matrix_col_rem(mat, 0);

    // Verify the new matrix dimensions and contents
    cr_assert_not_null(new_mat, "New matrix after row removal is NULL");
    cr_assert_eq(new_mat->num_rows, 3, "New matrix should have 3 rows");
    cr_assert_eq(new_mat->num_cols, 3, "New matrix should have 3 colsumns");
    double expected_values[9] = {2, 3, 4, 6, 7, 8, 10, 11, 12};
    for (unsigned int i = 0; i < 9; i++) {
        cr_assert_eq(((double *)new_mat->data)[i], expected_values[i], "Element [%u] is incorrect after column removal", i);
    }

    matrix_free(new_mat);
}


Test(matrix_init, col_remove_last) {

    matrix *mat = matrix_new(3, 4, sizeof(double));
    cr_assert_not_null(mat, "Matrix allocation returned NULL");

    // Initialize matrix with some values
    double *data = (double *)mat->data;
    for (unsigned int i = 0; i < 12; i++) {
        data[i] = i + 1; // 1 to 12
    }

    // Remove the last row
    matrix *new_mat = matrix_col_rem(mat, 3);

    // Verify the new matrix dimensions and contents
    cr_assert_not_null(new_mat, "New matrix after row removal is NULL");
    cr_assert_eq(new_mat->num_rows, 3, "New matrix should have 3 rows");
    cr_assert_eq(new_mat->num_cols, 3, "New matrix should have 3 columns");
    double expected_values[9] = {1, 2, 3, 5, 6, 7, 9, 10, 11};
    for (unsigned int i = 0; i < 9; i++) {
        cr_assert_eq(((double *)new_mat->data)[i], expected_values[i], "Element [%u] is incorrect after column removal: (double*)new_mat->data[i] = %lf and expected_value[i] = %lf", i, ((double*)new_mat->data)[i], expected_values[i]);
    }

    matrix_free(new_mat);
}
