
import ctypes
import os
from typing import List, Tuple, Optional

from config import SubleqConfig

class SubleqInterpreter:
    """Python wrapper for the SUBLEQ interpreter C library."""
    
    def __init__(
            self, 
            library_path: str = "./interpreters/subleq/libsubleq.so", 
            max_output_length: int = 10000, 
            max_iter: int = 1000000
        ):
        """
        Initialize the SUBLEQ interpreter.
        
        Args:
            library_path: Path to the compiled shared library.
                         If None, looks for 'libsubleq.so' (Linux/Mac) or 'subleq.dll' (Windows)
                         in the current directory.
        """
        if library_path is None:
            if os.name == 'nt':  # Windows
                library_path = './subleq.dll'
            else:  # Linux/Mac
                library_path = './libsubleq.so'
        
        self.max_output_length = max_output_length
        self.max_iter = max_iter

        self.lib = ctypes.CDLL(library_path)
        
        # Define the function signature
        # long* subleq_interpreter(const long *code, size_t code_length,
        #                          const long *input, size_t input_length,
        #                          size_t max_output_length, size_t max_iter,
        #                          size_t *output_count, int *interp_status,
        #                          long **final_mem_out, size_t *final_mem_len_out)
        self.lib.subleq_interpreter.argtypes = [
            ctypes.POINTER(ctypes.c_long),     # code
            ctypes.c_size_t,                   # code_length
            ctypes.POINTER(ctypes.c_long),     # input
            ctypes.c_size_t,                   # input_length
            ctypes.c_size_t,                   # max_output_length
            ctypes.c_size_t,                   # max_iter
            ctypes.POINTER(ctypes.c_size_t),   # output_count
            ctypes.POINTER(ctypes.c_int),      # interp_status
            ctypes.POINTER(ctypes.POINTER(ctypes.c_long)),  # final_mem_out
            ctypes.POINTER(ctypes.c_size_t)    # final_mem_len_out
        ]
        self.lib.subleq_interpreter.restype = ctypes.POINTER(ctypes.c_long)
        
        # Define free helpers
        self.lib.free_output.argtypes = [ctypes.POINTER(ctypes.c_long)]
        self.lib.free_output.restype = None

        self.lib.free_final_mem.argtypes = [ctypes.POINTER(ctypes.c_long)]
        self.lib.free_final_mem.restype = None
    
    def run(
            self, 
            code: List[int], 
            input_data: List[int] = None,
            max_output_length: Optional[int] = None,
            max_iter: Optional[int] = None
        ) -> Tuple[List[int], List[int], int]:
        """
        Run SUBLEQ code.
        
        Args:
            code: List of integers representing the SUBLEQ program
            input_data: List of input integers (optional)
            max_output_length: Maximum number of output values
            max_iter: Maximum number of iterations
            
        Returns:
            Tuple of (output_list, final_mem_state, status) where:
                - output_list is a list of output integers
                - final_mem_state is a list of output integers
                - status is 0 for success, -1 for error
                
        Raises:
            RuntimeError: If the interpreter encounters an error
        """
        if input_data is None:
            input_data = []
        
        if max_output_length is None:
            max_output_length = self.max_output_length
        if max_iter is None:
            max_iter = self.max_iter

        # Convert Python lists to C arrays
        code_array = (ctypes.c_long * len(code))(*code)
        input_array = (ctypes.c_long * len(input_data))(*input_data) if input_data else None
        
        # Output parameters
        output_count = ctypes.c_size_t()
        interp_status = ctypes.c_int()

        # NEW: final memory outputs
        final_mem_ptr = ctypes.POINTER(ctypes.c_long)()
        final_mem_len = ctypes.c_size_t(0)
        
        # Call the C function
        output_ptr = self.lib.subleq_interpreter(
            code_array,
            len(code),
            input_array,
            len(input_data),
            max_output_length,
            max_iter,
            ctypes.byref(output_count),
            ctypes.byref(interp_status),
            ctypes.byref(final_mem_ptr),
            ctypes.byref(final_mem_len)
        )
        
        # Convert outputs to Python lists
        output_list: List[int] = []
        if output_count.value > 0 and bool(output_ptr):
            output_list = [output_ptr[i] for i in range(output_count.value)]
        
        final_mem_list: List[int] = []
        if final_mem_len.value > 0 and bool(final_mem_ptr):
            final_mem_list = [final_mem_ptr[i] for i in range(final_mem_len.value)]
        
        # Free the C-allocated memory
        if bool(output_ptr):
            self.lib.free_output(output_ptr)
        if bool(final_mem_ptr):
            self.lib.free_final_mem(final_mem_ptr)
        
        return output_list, final_mem_list, interp_status.value


# def subleq(code: List[int], 
#            input_data: List[int],
#            cfg: SubleqConfig) -> Tuple[List[int], List[int], int]:
#     """
#     Convenience function to run SUBLEQ code without creating an interpreter object.
    
#     Args:
#         code: List of integers representing the SUBLEQ program
#         input_data: List of input integers (optional)
#         cfg: SubleqConfig with
#             max_output_length: Maximum number of output values
#             max_iter: Maximum number of iterations
#             library_path: Path to the shared library (optional)
        
#     Returns:
#         (output_list, final_mem_state, status)
#     """

#     interpreter = SubleqInterpreter(cfg.library_path)
#     return interpreter.run(code, input_data, cfg.max_output_length, cfg.max_iter)