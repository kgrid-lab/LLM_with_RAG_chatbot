import unittest

from cockroft_gault_cr_cl import cockroft_gault_cr_cl
from ckd_epi_gfr_2021 import ckd_epi_gfr_2021
from chadsvasc import chadsvasc
from mean_arterial_pressure import mean_arterial_pressure
from ascvd_2013 import ascvd_2013
from bmi import bmi
from bsa import bsa

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

    def test_ascvd_2013(self):
        self.assertAlmostEqual(ascvd_2013(55, False, "female", False, 213, 50, 120, False, "white"), 0.021, places=3)
        self.assertAlmostEqual(ascvd_2013(55, False, "female", False, 213, 50, 120, False, "african american"), 0.030, places=3)
        self.assertAlmostEqual(ascvd_2013(55, False, "male", False, 213, 50, 120, False, "white"), 0.054, places=3)
        self.assertAlmostEqual(ascvd_2013(55, False, "male", False, 213, 50, 120, False, "african american"), 0.061, places=3)

    def test_bmi(self):
        self.assertAlmostEqual(bmi(2, 80), 20, places=1)
        self.assertAlmostEqual(bmi(1.8, 75), 23.1, places=1)
        self.assertAlmostEqual(bmi(1.61, 72), 27.8, places=1)

    def test_bsa(self):
        self.assertAlmostEqual(bsa(180, 180), 3.00, places=2)
        self.assertAlmostEqual(bsa(216, 150), 3.00, places=2)
        self.assertAlmostEqual(bsa(200, 80), 2.11, places=2)
        self.assertAlmostEqual(bsa(180, 75), 1.94, places=2)
        self.assertAlmostEqual(bsa(161, 72), 1.79, places=2)

if __name__ == '__main__':
    unittest.main()