from math import e
from math import log

def ascvd_2013(age: int, dm: bool, sex: str, smoker: bool, total_cholesterol: int, hdl: int, sbp: int, htn_tx: bool, race: str) -> float:
    """
    Parameters:
    - age: The patient's age in years. This calculation is only valid for ages 40-75.
    - dm: True if the patient has diabetes. False otherwise.
    - sex: The patient's sex, either male or female.
    - smoker: True if the patient smokes.
    - total_cholesterol: The patient's total serum cholesterol in miligrams per deciliter.
    - hdl: The patient's serum high-density lipoprotein cholesterol in miligrams per decileter.
    - sbp: The patient's systolic blood pressure in mm Hg.
    - htn_tx: True if the patient is being treated for hypertension. False otherwise.
    - race: The patient's race, either white or african american.
    Returns: The patient's risk of developing their first myocardial infarction or stroke in the next 10 years.
    """
    coefficients = {}
    values = {}

    if race == "african american":
        if sex == "female":
            coefficients["ln_age"] = 17.114
            coefficients["ln_age_sq"] = 0
            coefficients["ln_tot_col"] = 0.940
            coefficients["ln_age_ln_tot_col"] = 0
            coefficients["ln_hdlc"] = -18.920
            coefficients["ln_age_ln_hdlc"] = 4.475
            coefficients["ln_tx_sbp"] = 29.291
            coefficients["ln_age_ln_tx_sbp"] = -6.432
            coefficients["ln_utx_sbp"] = 27.820
            coefficients["ln_age_ln_utx_sbp"] = -6.087
            coefficients["smoker"] = 0.691
            coefficients["ln_age_smoker"] = 0
            coefficients["diabetes"] = 0.874
            baseline_survival = 0.9533
            mean_score = 86.61
        else:
            coefficients["ln_age"] = 2.469
            coefficients["ln_age_sq"] = 0
            coefficients["ln_tot_col"] = 0.302
            coefficients["ln_age_ln_tot_col"] = 0
            coefficients["ln_hdlc"] = -0.307
            coefficients["ln_age_ln_hdlc"] = 0
            coefficients["ln_tx_sbp"] = 1.916
            coefficients["ln_age_ln_tx_sbp"] = 0
            coefficients["ln_utx_sbp"] = 1.809
            coefficients["ln_age_ln_utx_sbp"] = 0
            coefficients["smoker"] = 0.549
            coefficients["ln_age_smoker"] = 0
            coefficients["diabetes"] = 0.645
            baseline_survival = 0.8954
            mean_score = 19.54
    else:
        if sex == "female":
            coefficients["ln_age"] = -29.799
            coefficients["ln_age_sq"] = 4.884
            coefficients["ln_tot_col"] = 13.540
            coefficients["ln_age_ln_tot_col"] = -3.114
            coefficients["ln_hdlc"] = -13.578
            coefficients["ln_age_ln_hdlc"] = 3.149
            coefficients["ln_tx_sbp"] = 2.019
            coefficients["ln_age_ln_tx_sbp"] = 0
            coefficients["ln_utx_sbp"] = 1.957
            coefficients["ln_age_ln_utx_sbp"] = 0
            coefficients["smoker"] = 7.574
            coefficients["ln_age_smoker"] = -1.665
            coefficients["diabetes"] = 0.661
            baseline_survival = 0.9665
            mean_score = -29.18
        else:
            coefficients["ln_age"] = 12.344
            coefficients["ln_age_sq"] = 0
            coefficients["ln_tot_col"] = 11.853
            coefficients["ln_age_ln_tot_col"] = -2.664
            coefficients["ln_hdlc"] = -7.990
            coefficients["ln_age_ln_hdlc"] = 1.769
            coefficients["ln_tx_sbp"] = 1.797
            coefficients["ln_age_ln_tx_sbp"] = 0
            coefficients["ln_utx_sbp"] = 1.764
            coefficients["ln_age_ln_utx_sbp"] = 0
            coefficients["smoker"] = 7.837
            coefficients["ln_age_smoker"] = -1.795
            coefficients["diabetes"] = 0.658
            baseline_survival = 0.9144
            mean_score = 61.18

    if htn_tx:
        coefficients["ln_utx_sbp"] = 0
        coefficients["ln_age_ln_utx_sbp"] = 0
    else:
        coefficients["ln_tx_sbp"] = 0
        coefficients["ln_age_ln_tx_sbp"] = 0

    values["ln_age"] = log(age)
    values["ln_age_sq"] = pow(values["ln_age"], 2)
    values["ln_tot_col"] = log(total_cholesterol)
    values["ln_age_ln_tot_col"] = values["ln_age"] * values["ln_tot_col"]
    values["ln_hdlc"] = log(hdl)
    values["ln_age_ln_hdlc"] = values["ln_age"] * values["ln_hdlc"]
    values["ln_tx_sbp"] = log(sbp)
    values["ln_age_ln_tx_sbp"] = values["ln_age"] * values["ln_tx_sbp"]
    values["ln_utx_sbp"] = log(sbp)
    values["ln_age_ln_utx_sbp"] = values["ln_age"] * values["ln_utx_sbp"]
    
    if smoker:
        values["smoker"] = 1
    else:
        values["smoker"] = 0

    values["ln_age_smoker"] = values["ln_age"] * values["smoker"]

    if dm:
        values["diabetes"] = 1
    else:
        values["diabetes"] = 0

    score = sum((coefficients[term] * values[term] for term in coefficients.keys()))

    return 1 - pow(baseline_survival, pow(e, score - mean_score))