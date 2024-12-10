from cockroft_gault_cr_cl import cockroft_gault_cr_cl
from ckd_epi_gfr_2021 import ckd_epi_gfr_2021
from chadsvasc import chadsvasc

assert round(cockroft_gault_cr_cl("female", 40, 70, 0.8)) == 103

assert round(ckd_epi_gfr_2021("female", 60, 0.8)) == 84

assert round(ckd_epi_gfr_2021("male", 60, 0.8)) == 101

assert round(ckd_epi_gfr_2021("female", 60, 0.8, 0.75)) == 98

assert round(ckd_epi_gfr_2021("male", 60, 0.8, 0.75)) == 111