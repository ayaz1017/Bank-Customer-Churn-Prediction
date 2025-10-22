from flask import Flask, render_template, request
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    if request.method == "POST":
        try:
            # Get form inputs
            gender = request.form.get("gender")
            geography = request.form.get("geography")
            has_cr_card = int(request.form.get("has_cr_card"))
            is_active_member = int(request.form.get("is_active_member"))
            credit_score = float(request.form.get("credit_score"))
            age = float(request.form.get("age"))
            tenure = float(request.form.get("tenure"))
            balance = float(request.form.get("balance"))
            num_of_products = int(request.form.get("num_of_products"))
            estimated_salary = float(request.form.get("estimated_salary"))

            # Create dataframe for prediction
            input_df = pd.DataFrame([{
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_of_products,
                "HasCrCard": has_cr_card,
                "IsActiveMember": is_active_member,
                "EstimatedSalary": estimated_salary
            }])

            # Predict
            pipeline = PredictPipeline()
            pred = pipeline.predict(input_df)
            prediction = "Customer will Exit" if pred[0] == 1 else "Customer will Stay"

            # Render result.html with prediction
            return render_template("result.html", prediction=prediction)

        except Exception as e:
            error = f"Error occurred: {e}"
            return render_template("index.html", error=error)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
