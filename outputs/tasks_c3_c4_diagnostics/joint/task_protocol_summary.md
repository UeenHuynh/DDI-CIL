# Task Protocol Summary

## Notes

- Task files are built from actual class IDs, not `range(num_classes)`.
- Task 0 contains 38 classes; Tasks 1-7 contain 20 classes each.
- Rare-class counts use train split thresholds `<=5`, `<=10`, and `<=20`.

## Task Summary

| protocol | seed | task_id | num_classes | classes | train_samples_current | validation_samples_current | test_samples_current | train_samples_seen | validation_samples_seen | test_samples_seen | rare_classes_le_5 | rare_classes_le_10 | rare_classes_le_20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 0 | 0 | 178 | 212,121,189,129,136,140,139,6,147,124,112,119,145,60,217,128,171,151,173,203,44,208,134,11,196,184,41,178,62,104,167,108,117,201,127,191,179,47,30,131,157,175,200,186,125,137,156,72,17,135,46,130,73,207,174,1,14,188,142,97,162,180,77,164,215,49,2,170,3,190,185,12,176,55,42,59,183,160,198,25,146,153,43,57,149,22,10,38,111,29,65,209,27,48,213,9,45,197,54,120,24,21,192,7,194,20,181,5,126,187,118,116,63,155,19,148,122,161,4,216,182,143,16,68,32,74,199,40,169,144,204,75,172,26,214,23,177,166,70,50,123,71,163,28,58,138,56,195,133,202,69,13,36,64,53,66,15,110,206,211,152,34,141,114,113,8,218,86,150,165,159,115,158,67,31,210,37,132 | 520841 | 173614 | 173614 | 520841 | 173614 | 173614 | 9 | 20 | 45 |
