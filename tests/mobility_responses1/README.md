# ✴️ Tests of Mobility Responses 

This computes mobility responses of pytorch models and compares them with the
expected mobility responses predicted from analytic models.

## 🛠️ Performing Tests

To perform the test use 

```bash
run_test.sh
```

🔹Several tests are performed by computing for configurations of particles the
mlmod mobility responses.  This is compared with those of analytic mobilities
based on Oseen and RPY.

🔹The ``pytest`` package is used to organize the tests and reported outcomes.
While the scripts try to automatically install needed dependencies, you 
may need to adjust manually depending on your system.

