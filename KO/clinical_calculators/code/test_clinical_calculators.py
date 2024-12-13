import unittest

from cockroft_gault_cr_cl import cockroft_gault_cr_cl
from ckd_epi_gfr_2021 import ckd_epi_gfr_2021
from chadsvasc import chadsvasc
from mean_arterial_pressure import mean_arterial_pressure
from ascvd_2013 import ascvd_2013
from bmi import bmi
from bsa import bsa
from corr_ca_alb import corr_ca_alb
from wells import wells
from mdrd_gfr import mdrd_gfr

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

    def test_corr_ca_alb(self):
        self.assertAlmostEqual(corr_ca_alb(10, 4, 4), 10, places=1)
        self.assertAlmostEqual(corr_ca_alb(10, 4, 4.1), 10.1, places=1)
        self.assertAlmostEqual(corr_ca_alb(10.5, 4.8, 4), 9.9, places=1)
        self.assertAlmostEqual(corr_ca_alb(10.5, 4.8, 4.1), 9.9, places=1)
        self.assertAlmostEqual(corr_ca_alb(10.5, 2.8, 4), 11.5, places=1)
        self.assertAlmostEqual(corr_ca_alb(10.5, 2.8, 4.1), 11.5, places=1)
        self.assertAlmostEqual(corr_ca_alb(10.5, 2.8, 4.2), 11.6, places=1)

    def test_wells(self):
        self.assertEqual(wells(False, False, False, False, False, False, False), 0)
        self.assertEqual(wells(False, False, False, False, False, True, False), 1)
        self.assertEqual(wells(True, False, False, False, False, False, False), 3)
        self.assertEqual(wells(True, False, True, False, False, False, False), 4.5)
        self.assertEqual(wells(True, False, True, False, False, False, True), 5.5)
        self.assertEqual(wells(True, False, True, True, False, False, True), 7)
        self.assertEqual(wells(True, True, True, True, True, True, True), 12.5)

    def test_mdrd_gfr(self):
        self.assertAlmostEqual(mdrd_gfr("female", 60, 0.8), 73.2, places=1)
        self.assertAlmostEqual(mdrd_gfr("female", 60, 0.8, False), 73.2, places=1)
        self.assertAlmostEqual(mdrd_gfr("female", 60, 0.8, True), 88.7, places=1)
        self.assertAlmostEqual(mdrd_gfr("female", 45, 0.75), 83.6, places=1)
        self.assertAlmostEqual(mdrd_gfr("female", 45, 0.75, False), 83.6, places=1)
        self.assertAlmostEqual(mdrd_gfr("female", 45, 0.75, True), 101.3, places=1)
        self.assertAlmostEqual(mdrd_gfr("male", 60, 0.8), 98.6, places=1)
        self.assertAlmostEqual(mdrd_gfr("male", 60, 0.8, False), 98.6, places=1)
        self.assertAlmostEqual(mdrd_gfr("male", 60, 0.8, True), 119.5, places=1)
        self.assertAlmostEqual(mdrd_gfr("male", 45, 0.75), 112.6, places=1)
        self.assertAlmostEqual(mdrd_gfr("male", 45, 0.75, False), 112.6, places=1)
        self.assertAlmostEqual(mdrd_gfr("male", 45, 0.75, True), 136.5, places=1)

if __name__ == '__main__':
    unittest.main()