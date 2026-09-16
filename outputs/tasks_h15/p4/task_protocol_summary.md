# Task Protocol Summary

## Notes

- Task files are built from actual class IDs, not `range(num_classes)`.
- Task 0 contains 38 classes; Tasks 1-7 contain 20 classes each.
- Rare-class counts use train split thresholds `<=5`, `<=10`, and `<=20`.

## Task Summary

| protocol | seed | task_id | num_classes | classes | train_samples_current | validation_samples_current | test_samples_current | train_samples_seen | validation_samples_seen | test_samples_seen | rare_classes_le_5 | rare_classes_le_10 | rare_classes_le_20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constrained_mass_balanced | 0 | 0 | 38 | 124,158,48,42,165,152,29,209,60,171,128,41,200,186,70,20,181,133,183,44,215,214,2,218,17,36,195,162,120,155,129,97,118,179,151,148,62,143 | 24773 | 8260 | 8259 | 24773 | 8260 | 8259 | 0 | 4 | 10 |
| constrained_mass_balanced | 0 | 1 | 10 | 26,16,55,202,75,126,185,206,77,173 | 22436 | 7480 | 7478 | 47209 | 15740 | 15737 | 0 | 0 | 3 |
| constrained_mass_balanced | 0 | 2 | 10 | 137,212,198,74,201,213,159,49,182,111 | 75104 | 25033 | 25035 | 122313 | 40773 | 40772 | 2 | 2 | 2 |
| constrained_mass_balanced | 0 | 3 | 10 | 15,65,169,150,139,117,47,199,160,145 | 60701 | 20235 | 20234 | 183014 | 61008 | 61006 | 3 | 3 | 3 |
| constrained_mass_balanced | 0 | 4 | 10 | 10,63,194,203,57,163,114,59,142,122 | 25404 | 8469 | 8467 | 208418 | 69477 | 69473 | 0 | 3 | 3 |
| constrained_mass_balanced | 0 | 5 | 10 | 32,56,153,127,167,149,112,11,140,136 | 24774 | 8259 | 8258 | 233192 | 77736 | 77731 | 0 | 0 | 2 |
| constrained_mass_balanced | 0 | 6 | 10 | 7,184,1,40,210,172,123,187,161,175 | 24773 | 8258 | 8260 | 257965 | 85994 | 85991 | 0 | 0 | 3 |
| constrained_mass_balanced | 0 | 7 | 10 | 30,130,66,72,146,180,190,132,192,53 | 24774 | 8261 | 8257 | 282739 | 94255 | 94248 | 0 | 0 | 3 |
| constrained_mass_balanced | 0 | 8 | 10 | 14,50,211,31,141,21,197,19,174,208 | 22711 | 7569 | 7570 | 305450 | 101824 | 101818 | 0 | 0 | 2 |
| constrained_mass_balanced | 0 | 9 | 10 | 177,9,8,115,43,189,45,188,138,207 | 22436 | 7478 | 7477 | 327886 | 109302 | 109295 | 0 | 0 | 2 |
| constrained_mass_balanced | 0 | 10 | 10 | 13,67,73,34,170,217,113,176,119,135 | 35721 | 11905 | 11906 | 363607 | 121207 | 121201 | 0 | 3 | 3 |
| constrained_mass_balanced | 0 | 11 | 10 | 6,37,166,125,157,24,12,196,108,104 | 67052 | 22349 | 22351 | 430659 | 143556 | 143552 | 2 | 2 | 2 |
| constrained_mass_balanced | 0 | 12 | 10 | 58,46,27,68,38,64,54,3,204,110 | 22512 | 7503 | 7504 | 453171 | 151059 | 151056 | 0 | 0 | 2 |
| constrained_mass_balanced | 0 | 13 | 10 | 25,144,23,178,147,134,121,191,86,116 | 42898 | 14299 | 14298 | 496069 | 165358 | 165354 | 2 | 3 | 3 |
| constrained_mass_balanced | 0 | 14 | 10 | 28,164,69,131,71,156,5,216,22,4 | 24772 | 8256 | 8260 | 520841 | 173614 | 173614 | 0 | 0 | 2 |
