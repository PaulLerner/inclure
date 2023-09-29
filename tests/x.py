# -*- coding: utf-8 -*-

import unittest

from inclure.common import preproc
from inclure.x import sub


class XTester(unittest.TestCase):
    def assert_expected(self, text, expected):
        output = sub(preproc(text))
        self.assertTrue(output==expected, msg=f"expected '{text}' -> '{expected}', got '{output}'")
        
    def test_sub(self):
        for text in ["auteur.ice", "au.trice.teur", "au.teur.trice", "aut.eur.rice"]:
            self.assert_expected(text, "auteur")
        for text in ["auteur.ices", "au.trice.teurs", "au.trice.teur.s", "au.teur.trices", 
                     "au.teur.trice.s", "aut.eur.rice.s"]:
            self.assert_expected(text, "auteurs")
        for text in ["auteur", "1.2", "1,3", "autrices", "auteurs", "autrice", "religieuses."]:
            self.assert_expected(text, None)
    
    
if __name__ == '__main__':
    unittest.main()