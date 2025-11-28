import ctypes
import os
from typing import List, Tuple

class SubleqInterpreter:
    """Python wrapper for the SUBLEQ interpreter C library."""
    
    def __init__(self, library_path: str = None):
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
        
        self.lib = ctypes.CDLL(library_path)
        
        # Define the function signature
        # long* subleq_interpreter(const long *code, size_t code_length,
        #                          const long *input, size_t input_length,
        #                          size_t max_output_length, size_t max_iter,
        #                          size_t *output_count, int *interp_status)
        self.lib.subleq_interpreter.argtypes = [
            ctypes.POINTER(ctypes.c_long),  # code
            ctypes.c_size_t,                 # code_length
            ctypes.POINTER(ctypes.c_long),  # input
            ctypes.c_size_t,                 # input_length
            ctypes.c_size_t,                 # max_output_length
            ctypes.c_size_t,                 # max_iter
            ctypes.POINTER(ctypes.c_size_t), # output_count
            ctypes.POINTER(ctypes.c_int)     # interp_status
        ]
        self.lib.subleq_interpreter.restype = ctypes.POINTER(ctypes.c_long)
        
        # Define free_output signature
        self.lib.free_output.argtypes = [ctypes.POINTER(ctypes.c_long)]
        self.lib.free_output.restype = None
    
    def run(self, 
            code: List[int], 
            input_data: List[int] = None,
            max_output_length: int = 10000,
            max_iter: int = 1000000) -> Tuple[List[int], int]:
        """
        Run SUBLEQ code.
        
        Args:
            code: List of integers representing the SUBLEQ program
            input_data: List of input integers (optional)
            max_output_length: Maximum number of output values
            max_iter: Maximum number of iterations
            
        Returns:
            Tuple of (output_list, status) where:
                - output_list is a list of output integers
                - status is 0 for success, -1 for error
                
        Raises:
            RuntimeError: If the interpreter encounters an error
        """
        if input_data is None:
            input_data = []
        
        # Convert Python lists to C arrays
        code_array = (ctypes.c_long * len(code))(*code)
        input_array = (ctypes.c_long * len(input_data))(*input_data) if input_data else None
        
        # Output parameters
        output_count = ctypes.c_size_t()
        interp_status = ctypes.c_int()
        
        # Call the C function
        output_ptr = self.lib.subleq_interpreter(
            code_array,
            len(code),
            input_array,
            len(input_data),
            max_output_length,
            max_iter,
            ctypes.byref(output_count),
            ctypes.byref(interp_status)
        )
        
        # Convert output to Python list
        output_list = []
        if output_count.value > 0:
            output_list = [output_ptr[i] for i in range(output_count.value)]
        
        # Free the C-allocated memory
        self.lib.free_output(output_ptr)
        
        return output_list, interp_status.value


def subleq(code: List[int], 
               input_data: List[int] = None,
               max_output_length: int = 100000,
               max_iter: int = 1000000,
               library_path: str = None) -> Tuple[List[int], int]:
    """
    Convenience function to run SUBLEQ code without creating an interpreter object.
    
    Args:
        code: List of integers representing the SUBLEQ program
        input_data: List of input integers (optional)
        max_output_length: Maximum number of output values
        max_iter: Maximum number of iterations
        library_path: Path to the shared library (optional)
        
    Returns:
        Tuple of (output_list, status)
    """
    interpreter = SubleqInterpreter(library_path)
    return interpreter.run(code, input_data, max_output_length, max_iter)