import re


def pascal_case_to_snake_case(pascal: str) -> str:
    """
    Convert a CamelCase to snake_case.

    Args:
        pascal: The class or class name as a string.

    Returns:
        str: The snake_case version of the class name.
    """
    # Insert underscores before capital letters, except at the start
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', pascal)
    snake = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    return snake