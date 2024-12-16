def mean_arterial_pressure(systolic: int, diastolic: int) -> int:
    """
    Parameters:
    - systolic: The patient's systolic blood pressure in mm Hg.
    - diastolic: The patient's diastolic blood pressure in mm Hg.
    Returns: The patient's mean arterial pressure in mm Hg.
    """
    return round((systolic + (diastolic * 2)) / 3)