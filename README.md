Gemini
==============================

Segmenting prospective users.

Project Organization
------------

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   ├── raw            <- The original, immutable data dump.
    │   └── results        <- Prediction output.
    │
    ├── logs               <- Log files
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        └── models         <- Scripts to train models and then use trained models to make
            │                 predictions
            ├── predict_model.py
            └── train_model.py
         
Suggested Steps	to Reproduce	
--------

 Install required packages. 

    `pip install -r requirements.txt`

Navigate to /src/features and run build_features.py script. This will process the external data and export to data/interim.

	`python build_features.py`
	
Navigate to /src/models and run train_model.py. This will train and serialize a light gbm model.

	`python train_model.py`
    
Now run predict_model.py. This will output the predictions in the data/results folder.

	`python predict_model.py`

----------------------------------------    
    OR
----------------------------------------

   	 `make all`   
    


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
