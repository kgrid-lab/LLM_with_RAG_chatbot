import unittest

from cockroft_gault_cr_cl import cockroft_gault_cr_cl
from ckd_epi_gfr_2021 import ckd_epi_gfr_2021
from chadsvasc import chadsvasc
from mean_arterial_pressure import mean_arterial_pressure

class TestClinicalCalculators(unittest.TestCase):

    def test_cockroft_gault_cr_cl(self):
        self.assertEqual(round(cockroft_gault_cr_cl("female", 40, 70, 0.8)), 103)
        self.assertEqual(round(cockroft_gault_cr_cl("male", 40, 70, 0.8)), 122)

    def test_ckd_epi_gfr_2021(self):
        self.assertEqual(round(ckd_epi_gfr_2021("female", 60, 0.8)), 84)
        self.assertEqual(round(ckd_epi_gfr_2021("male", 60, 0.8)), 101)
        self.assertEqual(round(ckd_epi_gfr_2021("female", 60, 0.8, 0.75)), 98)
        self.assertEqual(round(ckd_epi_gfr_2021("male", 60, 0.8, 0.75)), 111)

    def test_chadsvasc(self):
        self.assertEqual(chadsvasc(62, "female", False, False, False, False, False), 1)
        self.assertEqual(chadsvasc(68, "female", False, False, False, False, False), 2)
        self.assertEqual(chadsvasc(68, "male", False, True, False, False, False), 2)

    def test_mean_arterial_pressure(self):
        self.assertEqual(mean_arterial_pressure(150, 90), 110)
        self.assertEqual(mean_arterial_pressure(110, 65), 80)
        self.assertEqual(mean_arterial_pressure(80, 50), 60)

if __name__ == '__main__':
    unittest.main()