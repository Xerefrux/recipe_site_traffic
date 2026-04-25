from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Named 'application' (not 'app') because AWS Elastic Beanstalk and
# some Render configurations expect this as the WSGI entry point name.
# The Procfile references it as: gunicorn application:application
application = Flask(__name__)
app = application  # convenience alias so `flask run` also finds it


CATEGORY_OPTIONS = [
    "Chicken", "Beverages", "Breakfast", "Dessert",
    "Lunch/Snacks", "Meat", "One Dish Meal", "Pork",
    "Potato", "Vegetable",
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html", categories=CATEGORY_OPTIONS)

    # POST — a user has submitted the prediction form.
    try:
        data = CustomData(
            calories=float(request.form.get("calories")),
            carbohydrate=float(request.form.get("carbohydrate")),
            sugar=float(request.form.get("sugar")),
            protein=float(request.form.get("protein")),
            category=request.form.get("category"),
            servings=int(request.form.get("servings")),
        )

        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        predictions, probabilities = predict_pipeline.predict(pred_df)

        # predictions[0] is 1 (High) or 0 (Not High)
        # probabilities[0] is P(High Traffic), expressed as a percentage
        result_label = "High Traffic" if predictions[0] == 1 else "Not High Traffic"
        confidence = round(float(probabilities[0]) * 100, 1)

        return render_template(
            "home.html",
            categories=CATEGORY_OPTIONS,
            results=result_label,
            confidence=confidence,
            # Echo submitted values back so the form retains its inputs
            submitted={
                "calories":     request.form.get("calories"),
                "carbohydrate": request.form.get("carbohydrate"),
                "sugar":        request.form.get("sugar"),
                "protein":      request.form.get("protein"),
                "category":     request.form.get("category"),
                "servings":     request.form.get("servings"),
            },
        )

    except Exception as e:
        return render_template(
            "home.html",
            categories=CATEGORY_OPTIONS,
            error=f"Prediction failed: {str(e)}. Please check your inputs.",
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)