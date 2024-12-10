def chadsvasc(age: int, sex: str, chf: bool, htn: bool, stroke: bool, vasc: bool, dm: bool) -> int:
    """
    Parameters:
    - age: The patient's age in years.
    - sex: The patient's sex, male or female.
    - chf: True if the patient has a history of heart failure. False otherwise.
    - htn: True if the patient has a history of hypertension. False otherwise.
    - stroke: True if the patient has a history of stroke, TIA, or thromboembolism. False otherwise.
    - vasc: True if the patient has a history of vascular disease, including prior myocardial infarction, peripheral arterial disease, or aortic plaque. False otherwise.
    - dm: True if the patient has a history of diabetes. False otherwise.
    Returns: A score representing the risk of stroke in a patient with atrial fibrillation. 0 is low-risk in males. 0 or 1 is low-risk in females. All other scores are higher risk.
    """
    score = 0

    if age >= 65:
        if age >= 75:
            score += 2
        else:
            score += 1

    if sex == "female":
        score += 1
    
    if htn:
        score += 1
    
    if stroke:
        score += 2
    
    if vasc:
        score += 1
    
    if dm:
        score +=1

    return score