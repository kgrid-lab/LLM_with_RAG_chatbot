def cockroft_gault_cr_cl(sex: str, age: int, weight: float, creatinine: float) -> float:
    """
    Parameters:
    - sex (str): Sex of the patient, male or female.
    - age (int): Age of the patient in years.
    - weight (float): Weight of the patient in kilograms.
    - creatinine (float): Serum creatinine concentration for the patient, in milligrams per deciliter.
    """
    if sex == "female":
        mult = 0.85
    else:
        mult = 1
    
    return (((140 - age) * weight) / (72 * creatinine)) * mult