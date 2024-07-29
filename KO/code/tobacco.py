def tobacco_qaly(smokeyear: int, quityear: int, cigperday: int, age: int, gender: int) -> float:
    """
    Parameters:
    - smokeyear (int): Number of yearssince the individual began smoking. 
    - quityear (int): Number of years since the individual quit smoking. If the person is currently smoking, this should be 0.
    - cigperday (int): Average number of cigarettes smoked per day.
    - age (int): Age of the individual in years.
    - gender (int): Gender of the individual (0 for female, 1 for male).
    """
    if smokeyear > 0 and quityear == 0 and cigperday > 0:
        x = (-0.0624067 * age
             - 1.314849 * (gender == 0)
             + 0.0112053 * age * (gender == 0)
             + 5.455188 * (cigperday < 15)
             + 8.53844 * (15 <= cigperday < 25)
             + 11.44078 * (cigperday >= 25)
             - 0.0502293 * age * (cigperday < 15)
             - 0.0791994 * age * (15 <= cigperday < 25)
             - 0.1073951 * age * (cigperday >= 25)
             + 2.765361 * (gender == 0) * (cigperday < 15)
             + 2.598462 * (gender == 0) * (15 <= cigperday < 25)
             + 5.994661 * (gender == 0) * (cigperday >= 25)
             - 0.027659 * age * (gender == 0) * (cigperday < 15)
             - 0.0260393 * age * (gender == 0) * (15 <= cigperday < 25)
             - 0.061088 * age * (gender == 0) * (cigperday >= 25)
             + 6.808624)
        return max(x, 1)
    else:
        return 0