As the guidelines request every file of the project must be edited mainly by one author only, here's a proposal on the division of the workload and folder structure:

seabed_classifier
_CMakeLists.txt
_include
__preprocessing.hpp   <-- (Member B)
__classifier.hpp      <-- (Member C)
__utils.hpp           <-- (Member A)
_src
__main.cpp            <-- (Member A)
__preprocessing.cpp   <-- (Member B)
__classifier.cpp      <-- (Member C)
__utils.cpp           <-- (Member A)
