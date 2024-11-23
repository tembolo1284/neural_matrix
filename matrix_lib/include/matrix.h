#ifndef MATRIX_H
#define MATRIX_H
#include <stdbool.h>
#include <stdlib.h>

typedef struct matrix_s {
  unsigned int num_rows;
  unsigned int num_cols;
  void *data;
  bool is_square;
} matrix;

typedef struct Range_s {
  int start;
  int end;
} Range;

typedef struct {
  matrix *L;
  matrix *U;
  matrix *P;
  unsigned int num_permutations;

} matrix_lup;

typedef struct {
  matrix *L;
  unsigned int num_permutations;
} matrix_cholesky;


/******* Matrix Initialization Operations *******/
matrix *matrix_new(unsigned int num_rows, unsigned int num_cols, size_t element_size);

matrix *matrix_rand(unsigned int num_rows, 
                    unsigned int num_cols, 
                    double min, 
                    double max, 
                    size_t element_size);

matrix *matrix_sqr(unsigned int size, size_t element_size);
matrix *matrix_eye(unsigned int size, size_t element_size, const void *identity_element);

void matrix_free(matrix *mat);
void matrix_print(const matrix *matrix);
void matrix_printf(const matrix *matrix, const char *d_fmt);

int matrix_eqdim(const matrix *m1, const matrix *m2);
int matrix_eq(const matrix *m1, const matrix *m2, double tolerance);
bool matrix_is_symmetric(matrix *mat);
bool matrix_is_posdef(matrix *mat);

matrix *matrix_col_get(const matrix *mat, unsigned int col_num);
matrix *matrix_row_get(const matrix *mat, unsigned int row_num);

double matrix_at(const matrix *mat, unsigned int i, unsigned int j);
matrix *matrix_slice(matrix *mat, Range row_range, Range col_range);
matrix *matrix_submatrix(const matrix *mat, Range row_range, Range col_range);
matrix *matrix_copy(const matrix *src);

void matrix_set(matrix *mat, unsigned int i, unsigned int j, double value);
void matrix_all_set(matrix *mat, const void *value, size_t value_size);
void matrix_diag_set(matrix *mat, const void *value, size_t value_size);

void matrix_transpose(matrix *mat);
matrix *matrix_stackv(const matrix *mat1, const matrix *mat2);
matrix *matrix_stackh(const matrix *mat1, const matrix *mat2);

/******* Internal Structure Change Functions *******/
matrix *matrix_row_rem(matrix *mat, unsigned int row);
matrix *matrix_col_rem(matrix *mat, unsigned int col);

void matrix_swap_rows(matrix *mat, unsigned int row1, unsigned int row2);
void matrix_swap_cols(matrix *mat, unsigned int col1, unsigned int col2);


/******* Internal Structure Change Functions *******/

/*******   Matrix math operations   *******/

double matrix_trace(matrix *mat);

//multiply a row with a scalar
void matrix_row_mult_r(matrix *mat, unsigned int row, double value);

//multiply a column with a scalar
void matrix_col_mult_r(matrix *mat, unsigned int col, double value);

//multiply a matrix with a scalar (all rows and columns)
void matrix_mult_r(matrix *mat, double value);

void matrix_row_addrow(matrix *mat, unsigned int row1_index, unsigned int row2_index, double factor);


matrix *matrix_add(const matrix *mat1, const matrix *mat2);
matrix *matrix_subtract(const matrix *mat1, const matrix *mat2);
matrix *matrix_mult(const matrix *mat1, const matrix *mat2);

int matrix_pivotidx(matrix *mat, unsigned int col, unsigned int row);
matrix *matrix_ref(matrix *mat);

matrix_lup *matrix_lup_new(matrix *L, matrix *U, matrix *P, unsigned int num_permutations);
matrix_lup *matrix_lup_solve(matrix *m);
void matrix_lup_free(matrix_lup *lu);

matrix *matrix_ls_solvefwd(matrix *L, matrix *b);
matrix *matrix_ls_solvebck(matrix *U, matrix *b);
matrix *matrix_ls_solve(matrix_lup *lu, matrix *b);

matrix *matrix_inv(matrix *mat);

double matrix_det(matrix_lup *lup);

matrix_lup *matrix_cholesky_solve(matrix *mat);
void matrix_cholesky_free(matrix_lup *cholesky);

#endif //MATRIX_H

