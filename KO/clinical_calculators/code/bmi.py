def bmi(height: float, weight: float) -> float:
    """
    Paramters:
    - height: The patient's height in meters.
    - weight: The patient's weight in kilograms.
    Returns: The Body Mass Index
    """
    return weight / pow(height, 2)