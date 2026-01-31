# importing needed libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# prepare the data

data = {

    "bedrooms": [2, 3, 4, 3, 5, 4, 3, 2, 4, 5],

    "size_sqft": [800, 1200, 1500, 1100, 2000, 1600, 1300, 900, 1400, 2100],

    "price": [150000, 200000, 250000, 180000, 300000, 270000, 220000, 160000, 240000, 320000]

}

df = pd.DataFrame(data)

print("First 5 rows of the dataset:")
print(df.head(), "\n")

# define features and target variable

X = df[["bedrooms", "size_sqft"]]
y = df["price"]

# split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# define regularization strengths

alphas = [0.01, 0.1, 1, 10]  # Regularization strengths for Ridge, Lasso, ElasticNet

# store results

results = {}
coefficients_dict = {}

# train models for each alpha

for alpha in alphas:

    print(f"\n--- Regularization strength: alpha = {alpha} ---\n")
    
    models = {
        "Linear Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]),
        "Ridge": Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=alpha))
        ]),
        "Lasso": Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Lasso(alpha=alpha))
        ]),
        "ElasticNet": Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', ElasticNet(alpha=alpha, l1_ratio=0.5))
        ])
    }
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)

        # store results

        results[(model_name, alpha)] = (mse_train, mse_test)
        coefficients_dict[(model_name, alpha)] = model.named_steps['regressor'].coef_ # Get coefficients
        intercept = model.named_steps['regressor'].intercept_ # Get intercept

        coefficients_dict [(model_name, alpha)] = model.named_steps['regressor'].coef_

        print(f"{model_name} | alpha={alpha} -> Train MSE: {mse_train:.2f}, Test MSE: {mse_test:.2f}")
        print(f"{model_name} Coefficients: {coefficients_dict[(model_name, alpha)]}")
        print(f"{model_name} Intercept: {intercept}\n")

    # plot traning and testing MSE for each model
    train_mse = {}
    test_mse = {}
    for model_name in ["Linear Regression", "Ridge", "Lasso", "ElasticNet"]:
        train_mse[model_name] = results[(model_name, alpha)][0]
        test_mse[model_name] = results[(model_name, alpha)][1]
    plt.figure(figsize=(10, 6))
    plt.yscale('log')
    plt.scatter(list(train_mse.keys()), list(train_mse.values()), marker='o', label=f"Train MSE")
    plt.scatter(list(test_mse.keys()), list(test_mse.values()), marker='x', label=f"Test MSE")
    plt.xlabel(f'Regularization Strength (alpha)={alpha}')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(f'Training vs Testing MSE for Linear, Ridge, Lasso, ElasticNet with alpha={alpha}')
    plt.grid(True, alpha=0.3)
    plt.bbox_inches = 'tight'
    plt.legend()
    plt.savefig(f'mse_plot_alpha_{alpha}.png')
    plt.show()
    plt.close()

    # plot coefficients shrinkage
plt.figure(figsize=(10, 6))
for model_name in ["Ridge", "Lasso", "ElasticNet"]:
        coefs = np.array([coefficients_dict[(model_name, a)] for a in alphas])
        for i in range(coefs.shape[1]):

         plt.plot(alphas, coefs[:, i], marker='o', label=f"{model_name} Coef {i+1}")
    
plt.xlabel('Regularization Strength (alpha)')
plt.yscale('log')
plt.ylabel('Coefficient Value')
plt.title('Coefficient Shrinkage for Ridge, Lasso, ElasticNet')
plt.grid(True, alpha=0.3)
plt.bbox_inches = 'tight'
plt.legend()
plt.savefig(' coefficients_plot.png')
plt.show()
plt.close()