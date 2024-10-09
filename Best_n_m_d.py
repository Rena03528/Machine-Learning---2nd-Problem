def tune_ridge_model(X_matrix, Y_matrix):
   
    alpha_values = np.arange(1e-5, 1, 0.01) 
    best_sse = float('inf')
    best_alpha = None
    best_model = None

    for value in alpha_values:
        ridge_model = Ridge(alpha=value).fit(X_matrix, Y_matrix.ravel())
        y_pred = ridge_model.predict(X_matrix)

        sse = np.sum((Y_matrix.ravel() - y_pred) ** 2)

        if sse < best_sse:
            best_sse = sse
            best_alpha = value
            best_model = ridge_model

    return best_model, best_alpha, best_sse

best_sse = float('inf')
best_params = None
best_model = None

for n in range(1, 10):
    for m in range(1, 10):
        for d in range(1, 10):
   
            X_matrix = X_matrix_creation(n, m, d, train_u, train_y)
            Y_matrix = Y_matrix_creation(n, m, d, train_y)

            model, alpha, sse = tune_ridge_model(X_matrix, Y_matrix)

            if sse < best_sse:
                best_alpha = alpha
                best_sse = sse
                best_params = (n, m, d)
                best_model = model

print(f"Best parameters (n, m, d): {best_params}")
print(f"Best SSE: {best_sse}")
