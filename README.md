# Employee Salary Predictor

A machine learning app built with **FastAPI** for model serving and **Streamlit** for an interactive user interface to predict employee salaries based on multiple input features like age, gender, job title, education level, and years of experience.

### Features:
- Predict employee salary using a **Linear Regression** model.
- Built with **FastAPI** for serving the model and **Streamlit** for the frontend UI.
- Data preprocessing includes **one-hot encoding** and **feature scaling** to improve model performance.

---

## 🛠️ Technologies Used

- **Python** (for all the logic)
- **Pandas** (data processing)
- **Scikit-learn** (linear regression model and preprocessing)
- **Streamlit** (frontend for user interaction)
- **FastAPI** (API to serve model predictions)
- **joblib** (model and scaler serialization)

---

## 🚀 Getting Started

### Prerequisites
Before you begin, ensure you have the following installed:

- Python 3.7+ 
- pip (Python package installer)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/employee-salary-predictor.git
   cd employee-salary-predictor


2. **Create and activate a virtual environment (optional but recommended)**
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`


3. **Install dependencies:**
pip install -r requirements.txt


📊 How the Model Works
The model predicts employee salaries based on the following features:

Age: The age of the employee.

Gender: The gender of the employee (Male/Female).

Education Level: The highest education level attained (High School, Bachelor's degree, Master's degree, or PhD).

Job Title: The job role of the employee (e.g., Manager, Engineer, Analyst, Administrator).

Years of Experience: Number of years the employee has worked.

The model is trained using linear regression, which is a simple algorithm for predicting numeric values based on input features.


🖥️ Running the App Locally
1. Run the FastAPI backend (optional for testing API directly):

From the project directory, run:
uvicorn app.main:app --reload

This will start the API on http://127.0.0.1:8000.

2. Run the Streamlit app:
streamlit run streamlit_app.py

This will launch the app at http://localhost:8501, where you can interact with the prediction model.


🌐 Deploying the App
The app is deployed online using Streamlit Cloud. Follow these steps to deploy the app on your own Streamlit Cloud:

Create a Streamlit account and log in.

Click on New App and link your GitHub repository.

Select the branch (main) and the file streamlit_app.py.

Streamlit will automatically install the required dependencies and deploy the app.

You will get a live link to share your app with others!


⚙️ File Structure
plaintext
Copy
Edit
employee-salary-predictor/
├── app/
│   ├── main.py             # FastAPI backend code for serving predictions
│   ├── model.pkl           # Trained Linear Regression model
│   ├── scaler.pkl          # Scaler for normalizing numerical features
│   └── features.pkl        # List of features used to train the model
├── data/
│   └── employee_data.csv   # Dataset used for training
├── notebooks/
│   └── 01_eda_and_modeling.ipynb  # Jupyter notebook with EDA and model training
├── requirements.txt        # Project dependencies
├── streamlit_app.py        # Streamlit frontend for salary prediction
├── .gitignore              # Git ignore file
└── README.md               # This README file


🤖 Model Evaluation
Model Type: Linear Regression

Evaluation Metrics:

R² Score: Measures the proportion of variance explained by the model.

RMSE (Root Mean Squared Error): Indicates the average error between predicted and actual salary.

MAE (Mean Absolute Error): Measures the average absolute difference between predicted and actual salary.


📄 License
This project is licensed under the MIT License – see the LICENSE file for details.


🙏 Acknowledgements
The dataset used for training the model is a simulated dataset designed to showcase the basic prediction of employee salaries.

Special thanks to the developers of Streamlit, FastAPI, Scikit-learn, and Pandas for their excellent libraries.

👨‍💻 Contributing
Feel free to fork the project, create issues, and submit pull requests. Contributions are always welcome!


### How to use:
1. **Copy** the above markdown text.
2. **Paste** it into a file called `README.md` in the root of your project directory.
3. **Commit and push** to GitHub to keep everything up-to-date.

Let me know if you need any other customizations or additions to your README!