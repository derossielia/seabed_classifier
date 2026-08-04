As the guidelines request every file of the project must be edited mainly by one author only, here's a proposal on the division of the workload and folder structure:

seabed_classifier/

├── CMakeLists.txt
├── include/
│   ├── preprocessing.hpp   <-- (Member B)
│   ├── classifier.hpp      <-- (Member C)
│   └── utils.hpp           <-- (Member A)
├── src/
│   ├── main.cpp            <-- (Member A)
│   ├── preprocessing.cpp   <-- (Member B)
│   ├── classifier.cpp      <-- (Member C)
│   └── utils.cpp           <-- (Member A)
