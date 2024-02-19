import pickle
import numpy as np
from flask import Flask, request, render_template
import gzip

#import pandas as pd





app = Flask(__name__)
with gzip.open("model.pickle","rb") as file:
    load_model = pickle.load(file)
@app.route('/')
def index():

    return render_template('index.html')

@app.route('/predict', methods = ['GET','POST'])
def predict():
 
        if request.method == 'POST':
            
           
        
           Annual_Income =float(request.form["Annual_Income"])
           Age = int(request.form["Age"])
           Num_Bank_Accounts =int(request.form["Num_Bank_Accounts"])
                      Num_Credit_Card = int(request.form["Num_Credit_Card"])
           Num_of_Loan = int(request.form["Num_of_Loan"])
           Interest_Rate = float(request.form["Interest_Rate"])
           Monthly_Balance = float(request.form["Monthly_Balance"])
           Total_EMI_per_month =float(request.form["Total_EMI_per_month"])
           Outstanding_Debt =float(request.form["Outstanding_Debt"])
           Delay_from_due_date = float(request.form["Delay_from_due_date"])
           Num_of_Delayed_Payment = float(request.form["Num_of_Delayed_Payment"])
           Changed_Credit_Limit =int(request.form["Changed_Credit_Limit"])
           Num_Credit_Inquiries = float(request.form["Num_Credit_Inquiries"])
           Credit_History_Age = int(request.form["Credit_History_Age"])
           Payment_of_Min_Amount = request.form["Payment_of_Min_Amount"]
           Payment_Behaviour =request.form["Payment_Behaviour"]
           Credit_Mix = request.form["Credit_Mix"]
           
           
           prediction = load_model.predict(np.array([[Annual_Income,Age,Num_Bank_Accounts,Num_Credit_Card,
                                              Num_of_Loan,Interest_Rate,Monthly_Balance,Total_EMI_per_month,
                                              Outstanding_Debt,Delay_from_due_date,Num_of_Delayed_Payment,
                                              Changed_Credit_Limit,Num_Credit_Inquiries,Credit_History_Age,
                                              Payment_Behaviour,Payment_of_Min_Amount,Credit_Mix]]))[0]
                                             
           output = int(prediction)
           if (output == 0):
         
               return render_template("result.html",prediction_text=" Credit Score is Good, predicted value is: ""{}".format(output))
           
           elif (output==1):
                return render_template("result.html",prediction_text="Credit Score is Poor, predicted value is: ""{}".format(output))
        
           else :
                return render_template("result.html",prediction_text= "Credit Score is Standard, predicted value is :""{}".format(output))
                                        
                                 
          

           
        # Handle the case where the input is not a valid float
        return "Invalid input. Please enter a valid value in the  fields."

if __name__ == '__main__':
    app.run(debug=True)
