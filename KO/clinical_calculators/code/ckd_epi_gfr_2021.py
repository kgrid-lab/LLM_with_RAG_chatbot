def ckd_epi_gfr_2021(sex: str, age: int, creatinine: float, cystatinc=None) -> float:
    """
    Parameters:
    - sex (str): Sex of the patient, male or female.
    - age (int): Age of the patient in years.
    - creatinine (float): Serum creatinine concentration for the patient, in milligrams per deciliter.
    - cystatinc (float): [optional] Serum cystatin-c concentration for the patient, in milligrams per liter.
    """
    if cystatinc is None:
        if sex == "female":
            mult = 1.012
            k = 0.7
            alpha = -0.241
        else:
            mult = 1
            k = 0.9
            alpha = -0.302
    
        return 142 * pow(min(creatinine / k, 1), alpha) * pow(max(creatinine / k, 1), -1.2) * pow(0.9938, age) * mult
    else:
        if sex == "female":
            mult = 0.963
            k = 0.7
            alpha = -0.219
        else:
            mult = 1
            k = 0.9
            alpha = -0.144
    
        return 135 * pow(min(creatinine / k, 1), alpha) * pow(max(creatinine / k, 1), -0.544) * pow(min(cystatinc / 0.8, 1), -0.323) * pow(max(cystatinc / 0.8, 1), -0.778) * pow(0.9961, age) * mult