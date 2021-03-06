def prediction_model(pclass, sex, age, sibsp, parch, fare, embarked, title):
    import pickle
    x = [[pclass, sex, age, sibsp, parch, fare, embarked, title]]
    randomforest = pickle.load(open('titanic.model.sav', 'rb'))
    prediction = randomforest.predict(x)
    if prediction == 0:
        prediction = 'Did not survive.'
    elif prediction == 1:
        prediction = 'Survided!'
    else:
        prediction = 'Error.'
    return prediction