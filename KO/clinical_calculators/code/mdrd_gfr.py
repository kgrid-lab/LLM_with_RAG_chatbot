from typing import Optional

def mdrd_gfr(sex: str, age: int, cr: float, race_black: Optional[bool]) -> float:
    """
    Parameters:
    - sex: The patient's sex, either male or female.
    - age: The patient's age in years.
    - cr: The patient's serum creatinine level in miligrams per deciliter.
    - race_black: Optional. True if the patient's race is Black. False otherwise.
    Returns: The patient's estimated glomerular filtration rate (GFR) in mL/min/1.73m^2.
    """
    if race_black:
        race_mult = 1.212
    else:
        race_mult = 1
    
    if sex == "female":
        sex_mult = 0.742
    else:
        sex_mult = 1

    return 175 * pow(cr, -1.154) * pow(age, -0.203) * sex_mult * race_mult