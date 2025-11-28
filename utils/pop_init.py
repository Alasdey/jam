import random
from typing import List


def random_code(length: int, 
                min_value: int, 
                max_value: int) -> List[int]:
    """
    Generate a random SUBLEQ program.
    
    Args:
        length: Number of integers in the program 
        min_value: Minimum value for random integers 
        max_value: Maximum value for random integers 
        
    Returns:
        List of random integers representing a SUBLEQ program
    """
    
    code = [random.randint(min_value, max_value) for _ in range(length)]
    
    return code


def random_population(pop_size: int,
                     code_length: int,
                     min_value: int,
                     max_value: int) -> List[List[int]]:
    """
    Generate a random population of SUBLEQ programs.
    
    Args:
        pop_size: Number of programs in the population
        code_length: Length of each program
        min_value: Minimum value for random integers
        max_value: Maximum value for random integers
        
    Returns:
        List of SUBLEQ programs
    """
    pop = [random_code(code_length, min_value, max_value) 
            for _ in range(pop_size)]
    
    return pop