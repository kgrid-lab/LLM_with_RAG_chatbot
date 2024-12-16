def bsa(height: float, weight: float) -> float:
    """
    Paramters:
    - height: The patient's height in centimeters.
    - weight: The patient's weight in kilograms.
    Returns: The patient's body surface area in square meters.
    """
    return pow(weight * height, 0.5) / 60