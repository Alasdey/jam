#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- Subleq Interpreter Implementation --- */

// Helper function to free the output array from Python
void free_output(long *output) {
    free(output);
}

// NEW: Helper to free the final memory array from Python
void free_final_mem(long *ptr) {
    free(ptr);
}

/*
 subleq_interpreter: Executes subleq code with bounded output and iteration count.
   - code: an array of longs representing the program (each 3 numbers is one instruction).
   - code_length: number of longs in the code array.
   - input: an array of input longs.
   - input_length: number of longs in the input array.
   - max_output_length: the maximum number of output longs allowed.
   - max_iter: the maximum number of iterations allowed.
   - output_count: pointer to a size_t that will hold the number of output longs produced.
   - interp_status: pointer to an int that will be set to 0 if interpretation finishes normally,
                    or -1 if an error occurs (out-of-bound access or max iterations reached).
   - final_mem_out: pointer to a long* that will receive a newly allocated copy
                    of the final memory state (length = code_length).
   - final_mem_len_out: pointer to size_t that will be set to code_length.
 Returns a dynamically allocated array of output longs (shrunk to the real output size).
*/
long* subleq_interpreter(const long *code, size_t code_length,
                            const long *input, size_t input_length,
                            size_t max_output_length, size_t max_iter,
                            size_t *output_count, int *interp_status,
                            long **final_mem_out, size_t *final_mem_len_out) {
    long *mem = malloc(code_length * sizeof(long));
    if (!mem) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    memcpy(mem, code, code_length * sizeof(long));

    long *output = malloc(max_output_length * sizeof(long));
    if (!output) {
        fprintf(stderr, "Memory allocation failed\n");
        free(mem);
        exit(1);
    }

    // Initialize new out parameters
    if (final_mem_out) *final_mem_out = NULL;
    if (final_mem_len_out) *final_mem_len_out = 0;

    size_t out_size = 0;
    size_t in_ptr = 0;
    size_t ip = 0;
    size_t iterations = 0;

    *interp_status = 0;  // assume no error initially

    while (ip < code_length) {
        // Check maximum iteration count.
        if (iterations >= max_iter) {
            *interp_status = -1;
            break;
        }
        iterations++;

        if (ip + 2 >= code_length)
            break;
        long a = mem[ip];
        long b = mem[ip + 1];
        long c = mem[ip + 2];

        if (c < 0)
            break;

        long operand;
        if (a < 0) {
            if (in_ptr < input_length) {
                operand = input[in_ptr++];
            } else {
                operand = 0;
            }
        } else {
            if ((size_t)a >= code_length) {
                *interp_status = -1;
                break;
            }
            operand = mem[a];
        }

        if (b < 0) {
            long result = 0 - operand;
            if (out_size >= max_output_length) {
                *interp_status = -1;
                break;
            }
            output[out_size++] = result;
            if (result <= 0) {
                ip = c;
                continue;
            }
            ip += 3;
        } else {
            if ((size_t)b >= code_length) {
                *interp_status = -1;
                break;
            }
            mem[b] = mem[b] - operand;
            if (mem[b] <= 0) {
                ip = c;
                continue;
            }
            ip += 3;
        }
    }

    // Capture final memory image before freeing internal buffer
    if (final_mem_out && final_mem_len_out) {
        long *final_mem = malloc(code_length * sizeof(long));
        if (!final_mem) {
            // If allocation fails, mark error but still return output
            *interp_status = -1;
        } else {
            memcpy(final_mem, mem, code_length * sizeof(long));
            *final_mem_out = final_mem;
            *final_mem_len_out = code_length;
        }
    }

    free(mem);

    // Only shrink the output array if at least one element was produced.
    if (out_size > 0) {
        long *final_output = realloc(output, out_size * sizeof(long));
        if (final_output != NULL) {
            output = final_output;
        }
    }
    *output_count = out_size;
    return output;
}
