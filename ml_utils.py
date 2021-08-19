from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


def train_XGB(X, y, X_cols, is_plot=False):
    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y)
    X_new = StandardScaler().fit_transform(X)
    y_new = StandardScaler().fit_transform(y)
    train_X, test_X, train_y, test_y = train_test_split(X_new, y_new, test_size=0.2, shuffle=False)
    my_model = XGBRegressor()
    my_model.fit(train_X, train_y)
    predictions = my_model.predict(test_X)
    # pred_proba = my_model.predict_proba(test_X)[:, 1]
    y_tag = scaler_y.inverse_transform(predictions)
    test_y = scaler_y.inverse_transform(test_y)
    # print("Accuracy : %.4g" % metrics.mean_squared_error(test_y, y_tag))
    # print("Accuracy : %.4g" % metrics.accuracy_score(test_y, y_tag))
    print("Mean absolute error =", round(metrics.mean_absolute_error(test_y, y_tag), 2))
    print("Mean squared error =", round(metrics.mean_squared_error(test_y, y_tag), 2))
    print("Median absolute error =", round(metrics.median_absolute_error(test_y, y_tag), 2))
    print("Explain variance score =", round(metrics.explained_variance_score(test_y, y_tag), 2))
    print("R2 score =", round(metrics.r2_score(test_y, y_tag), 2))
    """
    Mean absolute error: This is the average of absolute errors of all the data points in the given dataset.
    Mean squared error: This is the average of the squares of the errors of all the data points in the given dataset. It is one of the most popular metrics out there!
    Median absolute error: This is the median of all the errors in the given dataset. The main advantage of this metric is that it's robust to outliers. A single bad point in the test dataset wouldn't skew the entire error metric, as opposed to a mean error metric.
    Explained variance score: This score measures how well our model can account for the variation in our dataset. A score of 1.0 indicates that our model is perfect.
    R2 score: This is pronounced as R-squared, and this score refers to the coefficient of determination. This tells us how well the unknown samples will be predicted by our model. The best possible score is 1.0, but the score can be negative as well.
    """
    y_tag = y_tag.reshape(-1, 1)
    feat_imp = my_model.feature_importances_
    res_df = pd.DataFrame({'Features': X_cols, 'Importance': feat_imp}).sort_values(by='Importance',
                                                                                    ascending=False)
    if is_plot:
        res_df.plot('Features', 'Importance', kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()
    print(res_df)
    print(res_df["Features"].tolist())
    return my_model