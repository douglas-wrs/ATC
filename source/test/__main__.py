import unittest
import doctest

import source.test.text_categorization.init_helpers_test as tc_init_helpers_test
import source.test.text_categorization_test as tc_test

loader = unittest.TestLoader()
suite  = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(tc_init_helpers_test))
suite.addTests(loader.loadTestsFromModule(tc_test))
# suite.addTests(doctest.DocTestSuite(doctest.testfile("./text_categorization/init_helpers_doctest.txt")))

runner = unittest.TextTestRunner(verbosity=1)
result = runner.run(suite)