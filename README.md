<p align="center">
    <img src="static/asset/logo_whosapp.png" width="200" height="200" alt="Whosapp Logo">
</p>

# Whosapp
![UNISA - FIA](https://img.shields.io/badge/unisa-FIA-blue)
![Whatsapp Stilometry](https://img.shields.io/badge/whatsapp_stilometry-lightgreen)

Welcome to Whosapp! 🚀 This project was developed as part of the "Fundamentals of Artificial Intelligence" course at the _University of Salerno_. Our objective is to create a machine learning model capable of identifying the authors of WhatsApp chats. Read on to learn how to use the model! 📱

## Project Structure 🏗️

The project is organized into the following directories:

- `configs/`: Contains configuration files, including:
  - `configs/alias.json`: Alias configuration
  - `configs/config.json`: Feature configuration

- `data/`: Holds project data with subdirectories:
  - `data/rawdata/`: Raw data storage
  - `data/dataset/`: Dataset used for training
  - `data/wordlist/`: Wordlist used for processing

- `frontend/`: Contains the project's frontend components

- `logs/`: Stores project logs, categorized into sections

- `models/`: Houses the machine learning models of the project

- `src/`: Hosts the source code of the project

## Getting Started 🚀

### Prerequisites

1. Create the following folders:
    - `data/rawdata/`
    - `configs/`

2. In the `configs/` folder, create the following files:
    - `configs/alias.json` where you will put the alias configuration. The file must be in the following format:
    ```json
    {
        "Username": ["Alias1", "Alias2", "Alias3"],
        "Username2": ["Alias1", "Alias2", "Alias3"]
    }
    ```

### Demo Instructions 🛠️

1. Clone the repository and install the requirements:
   ```bash
   git clone https://github.com/danlig/WhosApp.git
   cd WhosApp
   pip install -r requirements.txt
    ```

2. Upload the chat you want to analyze in the `data/rawdata/` folder
3. Run the `py src/pipeline.py` script. For more information, run `py src/pipeline.py -h`.
4. Run the `py src/main.py` script to load and use the model
5. Finally, run `node frontend/index.js` to start the frontend of the project 
   
### Other Useful Commands 🛠️

- `py src/new_dataset.py`: Creates a new dataset from the raw data in `data/raw/` and saves it in `data/dataset/` in _.parquet_ format
- `py src/new_model.py`: Creates a new model from the dataset created with `py src/new_dataset.py` and saves it in `models/` in _.joblib_ format.
- `py src/pipeline.py`:  Creates a new dataset and model from the raw data in `data/raw/` and saves them in `data/dataset/` and `models/` respectively.
- `py src/test_features.py`: test the features that you have configured in the `configs/config.json` file
For more information on how to use these scripts, run `py <script_name>.py -h`
 
## Contributing 🤝

If you are interested in contributing to Whosapp, we welcome your collaboration! 
Feel free to contribute by following the guidelines outlined in our [CONTRIBUTING](CONTRIBUTING.md) page. 
Whether it's reporting issues, suggesting improvements, or submitting your own code changes, your contributions are highly appreciated.
Let's build Whosapp together! 🚀
---

### Authors
- [Daniele Liguori](https://github.com/danlig)
- [José Sgariglia](https://github.com/jose-sgariglia)
- [Luigi Turco](https://github.com/KronosPNG)










