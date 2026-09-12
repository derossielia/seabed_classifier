As the guidelines request every file of the project must be edited mainly by one author only, here's a proposal on the division of the workload and folder structure.

Member A (Elia De Rossi): include/utils.hpp, src/utils.cpp, include/preprocessing.hpp, src/preprocessing.cpp

Member B (Marco Rossi): include/classifier.hpp, src/classifier.cpp, src/main.cpp 

TASK DIVISION: 
    -   main.cpp extracts the ground truth label from the dataset folder structure, compares it to the classifier output, feeds the results to Utils, and prints the final table of metrics to the console.;
    -   utils performs the image loading, writes the text label files, implementsthe general math functions responsible of calculating evaluation metrics: Accuracy, Precision, Recall, and F1-score;
    -   preprocessing handles three actions on the original image in order to render it usable for classification: denoising, HSV conversion, contrast enhancement;
    -   classifier handles the specific task of assigning every image to its proper class via classical computer vision logic (performs prediction on the labels).
