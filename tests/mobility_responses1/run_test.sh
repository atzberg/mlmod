#!/bin/bash

# tell user the test to be performed 
echo "Perform tests for Oseen and RPY mobilities for the particle responses."

# perform the tests
python ./check_pytest.py
pytest ./test_response1.py  -vv -s

# remind user of the test performed 
echo "Finished testing Oseen and RPY mobilities for the particle responses."

