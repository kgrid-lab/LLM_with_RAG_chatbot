def qaly_value(age, gender, race, smokeyear, cigperday, quityear, totalcholesterol, hdl, sbp, htmedication, diabetes, alcohol_abuse):
    """  
    Parameters:
    age (int): Age of the individual in years. 
    gender (int): Gender of the individual (1 for male, 0 for female).
    race (int): Race of the individual (1 for white, 2 for black, and 3 for other).
    smokeyear (int): Number of years the individual has been smoking. 
    cigperday (int): Average number of cigarettes smoked per day.
    quityear (int): Number of years since the individual quit smoking. 
    totalcholesterol (float): Total cholesterol level in mg/dL. 
    hdl (float): High-density lipoprotein (HDL) cholesterol level in mg/dL. 
    sbp (float): Systolic blood pressure in mmHg. 
    htmedication (int): Hypertension medication status (1 if on medication, 0 if not).
    diabetes (int): Diabetes status (1 if diabetic, 0 if not).
    alcohol_abuse (int): Alcohol abuse status (1 if yes, 0 if no).    
    """
    if alcohol_abuse > 0:
        return (
            -0.0688071 * age
            + 0.0570005 * (gender == 1)
            - 0.1667344 * (race == 1)
            + 0.1383614 * (race == 2)
            - 0.0536393 * (race == 3)
            - 0.0257437 * ((smokeyear > 0) and (cigperday > 0) and (quityear == 0))
            - 0.00000938 * totalcholesterol
            + 0.0001198 * hdl
            - 0.0002621 * sbp
            + 0.0139843 * htmedication
            - 0.0316753 * diabetes
            + 5.446518
        )
    else:
        return 0
