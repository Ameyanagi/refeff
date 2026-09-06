"""Contracts for accepting repaired native reference products."""
from pathlib import Path
import runpy
import tempfile
import unittest

REPAIR = runpy.run_path(str(Path(__file__).with_name('repair-native-references.py')))


class ReferenceRepairContracts(unittest.TestCase):
    def test_requires_a_complete_fms_spectrum(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / 'phase.bin').write_text('1 2 2\n')
            (work / 'gg.bin').write_bytes(b'#SN# 1\n#SN# 2\n')
            (work / 'fms.bin').write_text('FMS\n2\n')
            good = '0 0 0 1 0.5 0.5\n1 1 1 0.9 0.5 0.4\n'
            (work / 'xmu.dat').write_text(good)
            self.assertEqual(REPAIR['validate_outputs'](work)['contour_points'], 2)
            for bad in ['0 0 0 1 1 0\n1 1 1 1 1 0\n', '0 0 0 nan 1 0\n', '0 0 0 1 0.5 0.5\n']:
                (work / 'xmu.dat').write_text(bad)
                with self.assertRaises(ValueError):
                    REPAIR['validate_outputs'](work)
            (work / 'xmu.dat').write_text(good)
            (work / 'gg.bin').write_bytes(b'#SN# 1\n')
            with self.assertRaises(ValueError):
                REPAIR['validate_outputs'](work)

    def test_unknown_native_source_is_rejected(self):
        for example in REPAIR['REPAIRS']:
            with self.assertRaises(ValueError):
                REPAIR['patched_source'](example, 'unexpected source')


if __name__ == '__main__':
    unittest.main()
