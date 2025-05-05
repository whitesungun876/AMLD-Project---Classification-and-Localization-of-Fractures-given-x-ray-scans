# AMLD Project - Classification and Localization of Fractures given x-ray scans

Final project for the course *Advanced Machine Learning for Data Science*.

Team members:
- Jieyu Lian
- Søren Mingon Esbensen
- Sebastian Faurby

## Instructions for this repository.

This chapter provides a walkthrough of the repository and some instructions in order to find the relevant scripts, data, and results for training, evaluating, and presenting our fracture classification and localization models.

1. **Presentation of the project:** The jupyter notebook called "AMLD - fracture classification and localization group.ipynb" in the folder **10. final presentation** contains a description and presentation of the whole project. This a redacted format of all the process that the project went through, from introduction and description of the central problem, domain and EDA to the training of the models and the presentation of their results. In the same folder one can find subfolders that contain some of the results, pictures and graphs described in the notebook. Therefore, the most important results can be found directly in the notebook, but feel free to look at the folders for additional results (it would be a very long notebook if we had taken every type of metric and plot into the final presentation).

2. **Scripts for training:** The scripts for training the models can be found in the folder *2. Train the models*. There are A LOT of scripts where we tried different things (thrown to _archive), but the ones that were used for the final results are:

    **faster RCNN**-folder:
* RCNN_train_noregularization.py: training of the Faster RCNN model without regularization
* RCNN_train_regularization.py: training of the Faster RCNN model with regularization
* RCNN_train_inference_reg.py: Inference and results of the Faster RCNN model with regularizaion
* RCNN_train_inference_reg.py: Inference and results of the Faster RCNN model without regularizaion
* RCNN_tuning*: three different scripts for tuning the Faster RCNN models. The optimized hyperparameters were used for the training scripts above.

    **yolo_models**-folder:
* YOLO11s-without-preprocessing.py: hypertuning and training of the YOLOv11s model
* YOLO11s-Predict.py: Inference and results of the YOLOv11s model

3. **Data**: We downloaded and placed the data in this repository. This can be found in the *Data* folder. The subfulder *FracAtlas* contains the original data as we got it from the authors of the original paper. The subfolder *split_data* contains the split of the data into training, validation and test parts. This data has also deleted some of the files that were corrupted (a very tiny part of the total data). All data extraction to train and test the models originated from this folder (split_data).

4. **Others**: We tried to clean the repository from unnecesary tests and files as much as we could. As you can see there are some other folders that are not described above. The contain either training scripts or results that were copied to the *10. final presentation* folder if necessary. Therefore, you can ignore this folders as they are not essential for the final results, but feel free to surf through them if you want. The file "requirements.txt" should contain most of the required packages but keep in mind that we worked from three different environments, so some might be missing.

### Other things to consider:
- The final weights of the Faster RCNN model were way too heavy to be pushed in the repository. They were therefore ignored. The weights of the YOLOv11s on the other hand, were small enough to be included. The notable difference of size was also discussed in the comparisson of the models in the jupyter notebook reffered in pt. 1 above.