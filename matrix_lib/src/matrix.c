#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

double matrix_rand_interval(double min, double max) {
    double d = (double) rand() / ((double) RAND_MAX + 1);
    return min + d * (max - min);
}

matrix *matrix_new(unsigned int num_rows, unsigned int num_cols, size_t element_size) {
    // Check for zero dimensions
    if (num_rows == 0 || num_cols == 0) {
        return NULL;
    }

    // Calculate total elements and check for overflow
    size_t total_elements;
    if (__builtin_mul_overflow(num_rows, num_cols, &total_elements)) {
        return NULL;
    }

    // Check if total allocation size would overflow
    size_t total_size;
    if (__builtin_mul_overflow(total_elements, element_size, &total_size)) {
        return NULL;
    }

    // Check if allocation is too large (arbitrary limit to prevent excessive memory usage)
    const size_t MAX_ALLOCATION = 1024 * 1024 * 1024; // 1GB limit
    if (total_size > MAX_ALLOCATION) {
        return NULL;
    }

    // Allocate matrix struct
    matrix *mat = calloc(1, sizeof(matrix));
    if (!mat) {
        return NULL;
    }

    // Allocate data array
    mat->data = calloc(total_elements, element_size);
    if (!mat->data) {
        free(mat);
        return NULL;
    }

    mat->num_rows = num_rows;
    mat->num_cols = num_cols;
    mat->is_square = (num_rows == num_cols);

    return mat;
}

matrix *matrix_rand(unsigned int num_rows, 
                    unsigned int num_cols, 
                    double min, double max, 
                    size_t element_size) {

    matrix *r = matrix_new(num_rows, num_cols, element_size); // Allocate new matrix
    if (!r) {
      return NULL;
    }

    double *data = (double *)r->data; // Cast to double* for direct access
    for (unsigned int i = 0; i < num_rows; i++) {
        for (unsigned int j = 0; j < num_cols; j++) {
            data[i * num_cols + j] = matrix_rand_interval(min, max);
        }
    }
    return r;
}


matrix *matrix_sqr(unsigned int size, size_t element_size) {
  return matrix_new(size, size, element_size);  
}

matrix *matrix_eye(unsigned int size, size_t element_size, const void *identity_element) {
    matrix *r = matrix_new(size, size, element_size);
    if (!r) {
        return NULL;
    }

    // Initialize all elements to 0
    memset(r->data, 0, size * size * element_size);

    // Set diagonal elements to the identity element
    for (unsigned int i = 0; i < size; i++) {
        memcpy((char*)r->data + (i * size + i) * element_size, identity_element, element_size);
    }

    return r;
}


void matrix_free(matrix *mat) {
  if (!mat) {
    return;
  }

  free(mat->data); 
  free(mat);
}

void matrix_print(const matrix *matrix) {
  matrix_printf(matrix, "%lf\t"); 
}

void matrix_printf(const matrix *matrix, const char *d_fmt) {
    if (!matrix || !matrix->data) {
        fprintf(stderr, "Cannot print NULL matrix or matrix with NULL data\n");
        return;
    }

    fprintf(stdout, "\n");
    
    for(unsigned int i = 0; i < matrix->num_rows; ++i) {
        for(unsigned int j = 0; j < matrix->num_cols; ++j) {
            double value = ((double *)(matrix->data))[i * matrix->num_cols + j];
            fprintf(stdout, d_fmt, value); 
        }
        fprintf(stdout, "\n");
    }
    
    fprintf(stdout, "\n");
}

int matrix_eqdim(const matrix *m1, const matrix *m2) {
  if(!m1 || !m2) return 0;  
  return (m1->num_cols == m2->num_cols) && (m1->num_rows == m2->num_rows);
}

int matrix_eq(const matrix *m1, const matrix *m2, double tolerance) {
    if (!m1 || !m2) return 0;

    if (!matrix_eqdim(m1, m2)) {
        printf("Matrices are not the same dimension!\n"); 
        return 0;
    }

    for (unsigned int i = 0; i < m1->num_rows; i++) {
        for (unsigned int j = 0; j < m1->num_cols; j++) {
            size_t element_size = sizeof(double); // Assuming elements are type double
            size_t index = i * m1->num_cols * element_size + j * element_size;
            double *elem1 = (double*)((char*)m1->data + index);
            double *elem2 = (double*)((char*)m2->data + index);
            
            if (fabs(*elem1 - *elem2) > tolerance) {
                printf("Mismatch in row %u and column %u\n", i, j); // Use %u for unsigned integers
                return 0;
            }
        }
    }

    return 1; // If we reach here, matrices are equal within the specified tolerance
}

double matrix_at(const matrix *mat, unsigned int i, unsigned int j) {
    if (!mat || !mat->data || i >= mat->num_rows || j >= mat->num_cols) {
        fprintf(stderr, "Invalid matrix access: matrix=%p, data=%p, i=%u, j=%u, num_rows=%u, num_cols=%u\n",
               (void*) mat, (void*)mat->data, i, j, mat->num_rows, mat->num_cols);
        exit(EXIT_FAILURE);
    }
        return ((double*)mat->data)[i * mat->num_cols + j];
}

bool matrix_is_symmetric(matrix *mat){
    if(!mat || !mat->data || !mat->is_square) {
        return false; // Non-square matrices are not symmetric
    }

    for (unsigned int i = 0; i < mat->num_rows; i++) {
        for(unsigned int j = 0; j < mat->num_cols; j++) {

            if(matrix_at(mat, i, j) != matrix_at(mat, j, i)) {
                return false; // Elements are not equal across diag, so matrix is not symmetric
            }
        }
    }
    return true; // All elements are equal when mirrored, mat is symmetric!
}


bool matrix_is_posdef(matrix *mat) {
    // Check if all leading principal minors (determinants of submatrices) are positive
    unsigned int n = mat->num_rows;
    matrix_lup *lup = matrix_lup_solve(mat);
    double determinant = 1.0;

    for (unsigned int i = 0; i < n; i++) {
        determinant *= matrix_at(lup->U, i, i);
        if (determinant <= 0) {
            matrix_lup_free(lup);
            return false;
        }
    }

    matrix_lup_free(lup);
    return true;
}


matrix *matrix_slice(matrix *mat, Range row_range, Range col_range) {
    // Check if all rows are needed
    if (row_range.start == -1) {
        if (col_range.start == -1) { // All rows, all columns
            return matrix_copy(mat);
        } else {
            return matrix_submatrix(mat, row_range, col_range);
        }
    } else { 
      // All other cases of single column or range of columns will just be a matrix_submatrix call
            return matrix_submatrix(mat, row_range, col_range);
    }
    
    printf("matrix slice not successfully executed.");
    return NULL;   
}


matrix *matrix_submatrix(const matrix *mat, Range row_range, Range col_range) {
     // Check for negative indices for start
    if (row_range.start < -1 || col_range.start < -1) {
        fprintf(stderr, "Invalid start index in range\n");
        return NULL;
    }

    // Check for negative indices for end, not including -1
    if ((row_range.end < -1) || (col_range.end < -1)) {
        fprintf(stderr, "Invalid end index in range\n");
        return NULL;
    }

    // Determine the start indices, handling the -1 case
    unsigned int row_start = (row_range.start == -1) ? 0 : row_range.start;
    unsigned int col_start = (col_range.start == -1) ? 0 : col_range.start;

    // Determine the end indices, handling the -1 case
    unsigned int row_end = (row_range.end == -1) ? (mat->num_rows) : (unsigned int)row_range.end;
    unsigned int col_end = (col_range.end == -1) ? (mat->num_cols) : (unsigned int)col_range.end;

    // Check if end indices exceed matrix dimensions
    if ((row_end - 1) > mat->num_rows || (col_end - 1) > mat->num_cols) {
        fprintf(stderr, "Range end exceeds matrix dimensions\n");
        return NULL;
    }

    // Calculate new dimensions
    unsigned int new_rows = (row_range.end == -1) ? row_end : row_end - row_start;
    unsigned int new_cols = (col_range.end == -1) ? col_end : col_end - col_start;    


    matrix *submat = matrix_new(new_rows, new_cols, sizeof(double));
    if (!submat) {
        fprintf(stderr, "Failed to allocate memory for submatrix\n");
        return NULL;
    }

    // Copy data from the original matrix to the submatrix
    for (unsigned int i = 0; i < new_rows; ++i) {
        for (unsigned int j = 0; j < new_cols; ++j) {
            ((double*)submat->data)[i * new_cols + j] = 
                ((double*)mat->data)[(row_start + i) * mat->num_cols + (col_start + j)];
        }
    }

    return submat;
}

void matrix_set(matrix *mat, unsigned int i, unsigned int j, double value) {
    if (mat == NULL || i >= mat->num_rows || j >= mat->num_cols) {
        // Check for invalid input or matrix dimensions
        return;  // Return without making any changes
    }

    // Calculate the index in the matrix data array for the specified element
    unsigned int index = i * mat->num_cols + j;

    // Update the value at the specified index
    ((double *)mat->data)[index] = value;
}


void matrix_all_set(matrix *mat, const void *value, size_t value_size) {
    for (unsigned int i = 0; i < mat->num_rows; ++i) {
        for (unsigned int j = 0; j < mat->num_cols; ++j) {
            unsigned int index = i * mat->num_cols + j;
            memcpy((char *)mat->data + index * value_size, value, value_size);
        }
    }
}

void matrix_diag_set(matrix *mat, const void *value, size_t value_size) {
    // Check if the matrix is square
    if (mat->num_rows != mat->num_cols) {
        fprintf(stderr, "Error: Matrix is not square.\n");
        return;
    }

    unsigned int min_dim = mat->num_rows;  // Since the matrix is square, num_rows == num_cols
    for (unsigned int i = 0; i < min_dim; ++i) {
        unsigned int index = i * mat->num_cols + i;
        memcpy((char *)mat->data + index * value_size, value, value_size);
    }
}

matrix *matrix_copy(const matrix *src) {
    if (src == NULL) {
        return NULL; // Handle null source matrix
    }

    // Manually allocate memory for the new matrix structure
    matrix *copy = malloc(sizeof(matrix));
    if (!copy) {
        return NULL; // Handle memory allocation failure
    }

    // Copy the matrix structure (excluding the data pointer)
    *copy = *src;

    // Allocate memory for the data array and copy the data
    copy->data = malloc(src->num_rows * src->num_cols * sizeof(double));
    if (!copy->data) {
        free(copy); // Clean up if data allocation fails
        return NULL;
    }
    memcpy(copy->data, src->data, src->num_rows * src->num_cols * sizeof(double));

    return copy;
}

void matrix_transpose(matrix *mat) {
    if (mat == NULL) {
        return; // Handle null matrix
    }

    // Create a new matrix to store the transpose
    matrix *transposed = matrix_new(mat->num_cols, mat->num_rows, sizeof(double));
    if (!transposed) {
        return; // Handle memory allocation failure
    }

    // Transpose the data
    for (unsigned int i = 0; i < mat->num_rows; ++i) {
        for (unsigned int j = 0; j < mat->num_cols; ++j) {
            ((double*)transposed->data)[j * mat->num_rows + i] = 
                ((double*)mat->data)[i * mat->num_cols + j];
        }
    }

    // Replace original matrix data with transposed data
    free(mat->data);
    mat->data = transposed->data;
    mat->num_rows = transposed->num_rows;
    mat->num_cols = transposed->num_cols;

    // Free the transposed matrix structure without freeing its data
    free(transposed);
}

matrix *matrix_stackv(const matrix *mat1, const matrix *mat2) {
    if (mat1 == NULL || mat2 == NULL) {
        return NULL; // Handle null matrices
    }

    // Check if both matrices have the same number of columns
    if (mat1->num_cols != mat2->num_cols) {
        fprintf(stderr, "Error: Matrices must have the same number of columns to stack.\n");
        return NULL;
    }

    // Create a new matrix to store the stacked matrices
    unsigned int new_rows = mat1->num_rows + mat2->num_rows;
    matrix *stacked = matrix_new(new_rows, mat1->num_cols, sizeof(double));
    if (!stacked) {
        return NULL; // Handle memory allocation failure
    }

    // Copy data from mat1 into the first part of stacked
    memcpy(stacked->data, mat1->data, mat1->num_rows * mat1->num_cols * sizeof(double));
    
    // Copy data from mat2 into the second part of stacked
    memcpy((char*)stacked->data + mat1->num_rows * mat1->num_cols * sizeof(double), 
           mat2->data, mat2->num_rows * mat2->num_cols * sizeof(double));

    return stacked;
}

matrix *matrix_stackh(const matrix *mat1, const matrix *mat2) {
    if (mat1 == NULL || mat2 == NULL) {
        return NULL; // Handle null matrices
    }

    // Check if both matrices have the same number of rows
    if (mat1->num_rows != mat2->num_rows) {
        fprintf(stderr, "Error: Matrices must have the same number of rows to stack horizontally.\n");
        return NULL;
    }

    // Create a new matrix to store the horizontally stacked matrices
    unsigned int new_cols = mat1->num_cols + mat2->num_cols;
    matrix *stacked = matrix_new(mat1->num_rows, new_cols, sizeof(double));
    if (!stacked) {
        return NULL; // Handle memory allocation failure
    }

    // Copy data from mat1 and mat2 into the stacked matrix
    for (unsigned int i = 0; i < mat1->num_rows; ++i) {
        // Copy data from mat1
        memcpy((char*)stacked->data + i * new_cols * sizeof(double),
            (char*)mat1->data + i * mat1->num_cols * sizeof(double),
            mat1->num_cols * sizeof(double));


        // Copy data from mat2
        memcpy((char*)stacked->data + (i * new_cols + mat1->num_cols) * sizeof(double), 
           (char*)mat2->data + i * mat2->num_cols * sizeof(double), 
           mat2->num_cols * sizeof(double));
    } 

    return stacked;

}

/*Matrix math operations*/

void matrix_row_mult_r(matrix *mat, unsigned int row, double value) {
    if (row >= mat->num_rows) {
        fprintf(stderr, "Row index out of bounds\n");
        return;
    }

    for (unsigned int j = 0; j < mat->num_cols; ++j) {
        ((double *)mat->data)[row * mat->num_cols + j] *= value;
    }
}

void matrix_col_mult_r(matrix *mat, unsigned int col, double value) {
    if (col >= mat->num_cols) {
        fprintf(stderr, "Column index out of bounds\n");
        return;
    }

    for (unsigned int i = 0; i < mat->num_rows; ++i) {
        ((double *)mat->data)[i * mat->num_cols + col] *= value;
    }
}

void matrix_mult_r(matrix *mat, double value) {
    for (unsigned int i = 0; i < mat->num_rows; ++i) {
        for (unsigned int j = 0; j < mat->num_cols; ++j) {
            ((double *)mat->data)[i * mat->num_cols + j] *= value;
        }
    }
}

void matrix_row_addrow(matrix *mat, unsigned int row1_index, unsigned int row2_index, double factor) {
    if (mat == NULL) {
        return; // Handle null matrix
    }

    // Check if row indices are within the bounds of the matrix
    if (row1_index >= mat->num_rows || row2_index >= mat->num_rows) {
        fprintf(stderr, "Error: Your matrix 1 or matrix 2 Row index selection is out of bounds.\n");
        return;
    }

    // Perform the row addition with the specified factor
    for (unsigned int i = 0; i < mat->num_cols; ++i) {
        double row1_value = ((double*)mat->data)[row1_index * mat->num_cols + i];
        double row2_value = ((double*)mat->data)[row2_index * mat->num_cols + i];
        ((double*)mat->data)[row2_index * mat->num_cols + i] = row2_value + (factor * row1_value);
    }
}


matrix *matrix_row_rem(matrix *mat, unsigned int row) {
    if (row >= mat->num_rows) {
        fprintf(stderr, "Row index out of bounds\n");
        return mat;
    }

    unsigned int new_rows = mat->num_rows - 1;
    matrix *new_mat = matrix_new(new_rows, mat->num_cols, sizeof(double));
    if (!new_mat) {
        return NULL; // Failed to allocate new matrix, return the original
    }

    double *new_data = (double *)new_mat->data;
    double *old_data = (double *)mat->data;

    for (unsigned int i = 0, new_i = 0; i < mat->num_rows; i++) {
        if (i == row) { 
            continue; // Skip the row to be removed
        }
        for (unsigned int j = 0; j < mat->num_cols; j++) {
            new_data[new_i * mat->num_cols + j] = old_data[i * mat->num_cols + j];
        }
        new_i++;
    }

    matrix_free(mat); // Free the old matrix
    return new_mat;   // Return the new matrix
}


matrix *matrix_col_rem(matrix *mat, unsigned int col) {
    if (col >= mat->num_cols) {
        fprintf(stderr, "Column index out of bounds\n");
        return mat;
    }

    unsigned int new_cols = mat->num_cols - 1;
    matrix *new_mat = matrix_new(mat->num_rows, new_cols, sizeof(double));
    if (!new_mat) {
        return NULL; // Failed to allocate new matrix, return the original
    }

    double *new_data = (double *)new_mat->data;
    double *old_data = (double *)mat->data;

    for (unsigned int i = 0; i < mat->num_rows; i++) {
    	for (unsigned int j = 0, new_j = 0; j < mat->num_cols; j++) {
        	if (j == col) {
            	    continue; // Skip the column to be removed
        	}
        	new_data[i * new_cols + new_j] = old_data[i * mat->num_cols + j];
                new_j++;
    	}
    }

    matrix_free(mat); // Free the old matrix
    return new_mat;   // Return the new matrix
}

void matrix_swap_rows(matrix *mat, unsigned int row1, unsigned int row2) {
    if (!mat || row1 >= mat->num_rows || row2 >= mat->num_rows) {
        // Handle invalid input
        fprintf(stderr, "Invalid input for matrix row swap.\n");
        return;
    }

    double *data = (double *)mat->data;
    for (unsigned int i = 0; i < mat->num_cols; i++) {
        double temp = data[row1 * mat->num_cols + i];
        data[row1 * mat->num_cols + i] = data[row2 * mat->num_cols + i];
        data[row2 * mat->num_cols + i] = temp;
    }

}

void matrix_swap_cols(matrix *mat, unsigned int col1, unsigned int col2) {
    if (!mat || col1 >= mat->num_cols || col2 >= mat->num_cols) {
        // Handle invalid input
        fprintf(stderr, "Invalid input for matrix column swap.\n");
        return;
    }

    double *data = (double *)mat->data;
    for (unsigned int i = 0; i < mat->num_rows; i++) {
        double temp = data[i * mat->num_cols + col1];
        data[i * mat->num_cols + col1] = data[i * mat->num_cols + col2];
        data[i * mat->num_cols + col2] = temp;
    }

}

double matrix_trace(matrix *mat) {
    double trace = 0.0;
    
    if(!mat || !mat->data || mat->num_rows != mat->num_cols) {
        return trace;
    }

    for (unsigned int i = 0; i < mat->num_rows; i++) {
        trace += *((double*)mat->data + i * mat->num_cols + i);
    }

    return trace;
}

matrix *matrix_add(const matrix *mat1, const matrix *mat2) {
    // Check if both matrices have the same dimensions
    if (!matrix_eqdim(mat1, mat2)) {
        fprintf(stderr, "Matrices dimensions do not match.\n");
        return NULL;
    }

    // Create a new matrix to store the result
    matrix *result = matrix_new(mat1->num_rows, mat1->num_cols, sizeof(double));
    if (!result) {
        return NULL; // Memory allocation failure
    }

    // Perform the addition
    double *data1 = (double *)mat1->data;
    double *data2 = (double *)mat2->data;
    double *result_data = (double *)result->data;

    for (unsigned int i = 0; i < mat1->num_rows; ++i) {
        for (unsigned int j = 0; j < mat1->num_cols; ++j) {
            result_data[i * mat1->num_cols + j] = data1[i * mat1->num_cols + j] + data2[i * mat1->num_cols + j];
        }
    }

    return result;
}

matrix *matrix_subtract(const matrix *mat1, const matrix *mat2) {
    // Check if both matrices have the same dimensions
    if (!matrix_eqdim(mat1, mat2)) {
        fprintf(stderr, "Matrices dimensions do not match.\n");
        return NULL;
    }

    // Create a new matrix to store the result
    matrix *result = matrix_new(mat1->num_rows, mat1->num_cols, sizeof(double));
    if (!result) {
        return NULL; // Memory allocation failure
    }

    // Perform the addition
    double *data1 = (double *)mat1->data;
    double *data2 = (double *)mat2->data;
    double *result_data = (double *)result->data;

    for (unsigned int i = 0; i < mat1->num_rows; ++i) {
        for (unsigned int j = 0; j < mat1->num_cols; ++j) {
            result_data[i * mat1->num_cols + j] = data1[i * mat1->num_cols + j] - data2[i * mat1->num_cols + j];
        }
    }

    return result;
}


matrix *matrix_mult(const matrix *mat1, const matrix *mat2) {
    // Check if the number of columns in mat1 equals the number of rows in mat2
    if (mat1->num_cols != mat2->num_rows) {
        fprintf(stderr, "Matrix dimensions are not compatible for multiplication.\n");
        return NULL;
    }

    // Create a new matrix to store the result
    matrix *result = matrix_new(mat1->num_rows, mat2->num_cols, sizeof(double));
    if (!result) {
        return NULL; // Memory allocation failure
    }

    // Perform matrix multiplication
    double *data1 = (double *)mat1->data;
    double *data2 = (double *)mat2->data;
    double *result_data = (double *)result->data;

    for (unsigned int i = 0; i < mat1->num_rows; ++i) {
        for (unsigned int j = 0; j < mat2->num_cols; ++j) {
            double sum = 0.0;
            for (unsigned int k = 0; k < mat1->num_cols; ++k) {
                sum += data1[i * mat1->num_cols + k] * data2[k * mat2->num_cols + j];
            }
            result_data[i * mat2->num_cols + j] = sum;
        }
    }

    return result;
}


int matrix_pivotidx(matrix *mat, unsigned int col, unsigned int row) {
    int i, maxi;
    double maxcol;
    double max = fabs(matrix_at(mat, row, col));
    maxi = row;
    for (i = row; i <(int)(mat->num_rows); i++) {
        maxcol = fabs(matrix_at(mat, i, col));
        if (maxcol > max) {
            max = maxcol;
            maxi = i;
        }
    }
    return (max < 1e-10) ? -1 : maxi; // You can adjust the tolerance as needed
}


matrix *matrix_ref(matrix *mat) {
    matrix *result = matrix_copy(mat);
    unsigned int i, j, k;
    int pivot;
    j = 0, i = 0;

    while (j < result->num_cols && i < result->num_cols) {
        // Find the pivot - the first non-zero entry in the column 'j' below row 'i'
        pivot = matrix_pivotidx(result, j, i);

        if (pivot < 0) {
            // All elements in the column are zeros
            // Move to the next column without doing anything
            j++;
            continue;
        }

        // Interchange rows, moving the pivot to the first row that doesn't have a pivot
        if (pivot != (int)i) {
            matrix_swap_rows(result, i, pivot);
        }

        // Multiply each element in the pivot row by the inverse of the pivot
        double pivot_value = matrix_at(result, i, j);
        matrix_row_mult_r(result, i, 1.0 / pivot_value);

        // Add multiples of the pivot row to make every element in the column below the pivot equal to 0
        for (k = i + 1; k < result->num_rows; k++) {
            double factor = -matrix_at(result, k, j);
            matrix_row_addrow(result, i, k, factor);
        }

        i++;
        j++;
    }

    return result;
}

void matrix_lup_free(matrix_lup *lu) {
    matrix_free(lu->P);
    matrix_free(lu->L);
    matrix_free(lu->U);
    free(lu);
}


// Function to create a new LUP decomposition
matrix_lup *matrix_lup_new(matrix *L, matrix *U, matrix *P, unsigned int num_permutations) {
    matrix_lup *lup = malloc(sizeof(matrix_lup));
    if (!lup) {
        fprintf(stderr, "Failed to allocate memory for LUP decomposition.\n");
        return NULL;
    }

    lup->L = L;
    lup->U = U;
    lup->P = P;
    lup->num_permutations = num_permutations;

    return lup;
}

// Function to perform LUP decomposition on a matrix
matrix_lup *matrix_lup_solve(matrix *m) {
    if (!m->is_square) {
        fprintf(stderr, "Matrix must be square for LUP decomposition.\n");
        return NULL;
    }

    // Create matrices for L, U, and P
    double identity_element = 1.0;
    matrix *L = matrix_eye(m->num_rows, sizeof(double), &identity_element);
    matrix *U = matrix_copy(m);
    matrix *P = matrix_eye(m->num_rows, sizeof(double), &identity_element);

    unsigned int num_permutations = 0;
    double mult;

    for (unsigned int j = 0; j < U->num_cols; j++) {
        // Find the pivot (maximum absolute value) in the current column
        unsigned int pivot = j;
        for (unsigned int i = j + 1; i < U->num_rows; i++) {
            if (fabs(matrix_at(U, i, j)) > fabs(matrix_at(U, pivot, j))) {
                pivot = i;
            }
        }

        if (fabs(matrix_at(U, pivot, j)) < 1e-10) {
            fprintf(stderr, "Matrix is degenerate, LUP decomposition failed.\n");
            matrix_free(L);
            matrix_free(U);
            matrix_free(P);
            return NULL;
        }

        if (pivot != j) {
            // Swap rows in U, L, and P to move the pivot to the current row
            matrix_swap_rows(U, j, pivot);
            matrix_swap_rows(L, j, pivot);
            matrix_swap_rows(P, j, pivot);
            num_permutations++;
        }

        for (unsigned int i = j + 1; i < U->num_rows; i++) {
            mult = matrix_at(U, i, j) / matrix_at(U, j, j);
            // Update U and store the multiplier in L
            matrix_row_addrow(U, j, i, -mult);
            matrix_set(L, i, j, mult);
        }
    }

    double val = 1.0;
    matrix_diag_set(L, &val, sizeof(double));

    // Create the LUP decomposition
    matrix_lup *lup = matrix_lup_new(L, U, P, num_permutations);
    return lup;
}

// Function to perform forward substitution to solve the linear system L * x = b
matrix *matrix_ls_solvefwd(matrix *L, matrix *b) {
    if (L == NULL || b == NULL) {
        fprintf(stderr, "Invalid input matrices for forward substitution.\n");
        return NULL;
    }

    if (L->num_rows != L->num_cols) {
        fprintf(stderr, "Matrix L must be square for forward substitution.\n");
        return NULL;
    }

    if (L->num_rows != b->num_rows || b->num_cols != 1) {
        fprintf(stderr, "Matrix dimensions are not compatible for forward substitution.\n");
        return NULL;
    }

    unsigned int n = L->num_rows;
    matrix *x = matrix_new(n, 1, sizeof(double));

    if (x == NULL) {
        fprintf(stderr, "Memory allocation failed for the solution vector.\n");
        return NULL;
    }

    // Perform forward substitution
    for (unsigned int i = 0; i < n; i++) {
        double sum = 0.0;
        for (unsigned int j = 0; j < i; j++) {
            sum += matrix_at(L, i, j) * matrix_at(x, j, 0);
        }
        double bi = matrix_at(b, i, 0);
        double xi = (bi - sum) / matrix_at(L, i, i);
        matrix_set(x, i, 0, xi);
    }

    return x;
}

matrix *matrix_ls_solvebck(matrix *U, matrix *b) {
    matrix *x = matrix_new(U->num_cols, 1, sizeof(double));
    int i = U->num_cols - 1;
    int j;
    double tmp;
    
    while (i >= 0) {
        tmp = ((double *)b->data)[i * b->num_cols + 0];
        for (j = i + 1; j < (int)(U->num_cols); j++) {
    	    tmp -= matrix_at(U, i, j) * matrix_at(x, j, 0);
        }
	matrix_set(x, i, 0, tmp / matrix_at(U, i, i));
        i--;
    }
    
    return x;
}

matrix *matrix_ls_solve(matrix_lup *lu, matrix *b) {
  // Check if dimensions are valid
  if (lu->U->num_rows != b->num_rows || b->num_cols != 1) {
    fprintf(stderr, "Dimensions of matrix are not valid.\n");
    return NULL;
  }

  // Calculate Pb = P*b
  matrix *Pb = matrix_mult(lu->P, b);

  // Solve L*y = Pb using forward substitution
  matrix *y = matrix_ls_solvefwd(lu->L, Pb);

  // Solve U*x = y using backward substitution
  matrix *x = matrix_ls_solvebck(lu->U, y);

  // Free intermediate matrices
  matrix_free(y);
  matrix_free(Pb);

  return x;
}


matrix *matrix_inv(matrix *mat) {
    if (!mat->is_square) {
        fprintf(stderr, "Matrix must be square for inversion.\n");
        return NULL;
    }

    // Perform LU decomposition
    matrix_lup *lu = matrix_lup_solve(mat);

    if (!lu) {
        fprintf(stderr, "LU decomposition failed. The matrix might be singular.\n");
        return NULL;
    }

    // Create an identity matrix of the same size as the input matrix
    double identity_element = 1.0;
    matrix *identity = matrix_eye(mat->num_rows, sizeof(double), &identity_element);

    // Initialize the inverse matrix
    matrix *inverse = matrix_new(mat->num_rows, mat->num_cols, sizeof(double));

    // Solve for the inverse by applying LU decomposition to each column of the identity matrix
    Range all_rows = {-1, -1};
    for (unsigned int col = 0; col < mat->num_cols; col++) {
        Range col_range = {col, col + 1};
        matrix *b = matrix_slice(identity, all_rows, col_range);  // Take single col from identity matrix
        matrix *x = matrix_ls_solve(lu, b);  // Solve for the column of the inverse

        // Copy the solution (x) to the corresponding column of the inverse matrix
        for (unsigned int row = 0; row < mat->num_rows; row++) {
            matrix_set(inverse, row, col, matrix_at(x, row, 0));
        }

        // Free the temporary matrices
        matrix_free(b);
        matrix_free(x);
    }

    // Free the LU decomposition and intermediate matrices
    matrix_lup_free(lu);
    matrix_free(identity);

    return inverse;

}

double matrix_det(matrix_lup *lup) {
    int k;
    int sign = (lup->num_permutations % 2 == 0) ? 1 : -1;
    matrix *U = lup->U;
    double product = 1.0;

    for(k = 0; k < (int)(U->num_rows); k++) {
        product *= matrix_at(U, k, k);
    }
    return product * sign;

}

// Function to perform Cholesky decomposition
matrix_lup *matrix_cholesky_solve(matrix *mat) {
    // Check if the input matrix is square and symmetric
    if (!mat->is_square || !matrix_is_symmetric(mat)) {
        fprintf(stderr, "Matrix must be square and symmetric for Cholesky decomposition.\n");
        return NULL;
    }

    unsigned int n = mat->num_rows;
    matrix *L = matrix_new(n, n, sizeof(double));

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j <= i; j++) {
            double sum = 0.0;
            if (j == i) {
                for (unsigned int k = 0; k < j; k++) {
                    double L_kk = matrix_at(L, j, k);
                    sum += L_kk * L_kk;
                }
                double L_ii = sqrt(matrix_at(mat, i, i) - sum);
                matrix_set(L, i, i, L_ii);
            } else {
                for (unsigned int k = 0; k < j; k++) {
                    sum += matrix_at(L, i, k) * matrix_at(L, j, k);
                }
                double L_ij = (matrix_at(mat, i, j) - sum) / matrix_at(L, j, j);
                matrix_set(L, i, j, L_ij);
            }
        }
    }

    matrix_lup *cholesky = (matrix_lup *)malloc(sizeof(matrix_lup));
    cholesky->L = L;
    cholesky->U = NULL; // Cholesky decomposition only requires L
    cholesky->P = NULL; // No permutation matrix for Cholesky
    cholesky->num_permutations = 0;

    return cholesky;
}

// Function to free memory allocated for Cholesky decomposition
void matrix_cholesky_free(matrix_lup *cholesky) {
    if (cholesky) {
        matrix_free(cholesky->L);
        free(cholesky);
    }
}
