// tests/layer_test.c
#include <criterion/criterion.h>
#include <criterion/logging.h>
#include "../include/layer.h"

Test(layer_init, create_layer) {
    unsigned int input_dim = 10;
    unsigned int output_dim = 5;
    
    layer* l = layer_new(input_dim, output_dim);
    
    cr_assert_not_null(l, "Layer creation failed");
    cr_assert_eq(l->input_dim, input_dim, "Input dimension mismatch");
    cr_assert_eq(l->output_dim, output_dim, "Output dimension mismatch");
    cr_assert_null(l->forward, "Forward function should be NULL initially");
    cr_assert_null(l->backward, "Backward function should be NULL initially");
    cr_assert_null(l->update, "Update function should be NULL initially");
    cr_assert_null(l->free, "Free function should be NULL initially");
    
    layer_free(l);
}

Test(layer_init, invalid_dimensions) {
    layer* l1 = layer_new(0, 5);
    cr_assert_null(l1, "Layer creation should fail with zero input dimension");
    
    layer* l2 = layer_new(5, 0);
    cr_assert_null(l2, "Layer creation should fail with zero output dimension");
}
