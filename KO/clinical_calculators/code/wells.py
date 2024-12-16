def wells(clin_sx_dvt: bool, pe_1_dx: bool, hr_gt_100: bool, immob_surg: bool, prev_dx: bool, hemoptysis: bool, malignancy: bool) -> float:
    """
    Parameters:
    - clin_sx_dvt: True if the patient has clinical signs or symptoms of DVT (deep vein thrombosis). False otherwise.
    - pe_1_dx: True if pulmonary embolism is the leading diagnosis or equally likely as another diagnosis. False otherwise.
    - hr_gt_100: True if the patient's heart rate is greater than 100 beats per minute. False otherwise.
    - immob_surg: True if the patient has been immobilized for at least 3 days or has had surgery in the previous 4 weeks. False otherwise.
    - prev_dx: True if the patient has previously been objectively diagnosed with DVT or pulmonary embolism. False otherwise.
    - hemoptysis: True if the patient is experiencing hemoptysis. False otherwise.
    - malignancy: True if the patient has a malignancy that has been treated within the past 6 months or has received palliative care. False otherwise.
    Returns: A score representing a non-pregnant adult patient's probability of having a pulmonary embolism (PE) in the emergency department.
    Scores less than 2 represent a low risk of PE. Scores from 2 to 6 represent a moderate risk of PE. Scores greater than 6 represent a high risk of PE.
    At scores greater than 4, a d-dimer test cannot rule out a PE.
    """
    score = 0
    if clin_sx_dvt:
        score += 3
    if pe_1_dx:
        score += 3
    if hr_gt_100:
        score += 1.5
    if immob_surg:
        score += 1.5
    if prev_dx:
        score += 1.5
    if hemoptysis:
        score += 1
    if malignancy:
        score += 1
    
    return score