def create_phi_test(i, n, d, m, y_test, test_u):

    phi = []

    y_end_index = i - 1
    y_initial_index = i - n

    x_end_index = i - d
    x_initial_index = i - d - m

    for j in range(y_initial_index, y_end_index + 1):
        if j > 0:
            phi.append(y_test[j])
        else:
            phi.append(0)

    phi = phi[::-1] 

    for j in range (x_end_index + 1, x_initial_index,-1):
        if j >= 0:
            phi.append(test_u[j])
        else:
            phi.append(0)

    return phi

def y_predict_generator(model, X_matrix, n, m, d, test_u):
    
    N = len(test_u)
    theta_matrix = model.coef_

    y_test = [] 

    for i in range(0,N-1):

        if i < d:
            y_test.append(0)
        else:
            phi_test_vector = create_phi_test(i, n, d, m, y_test, test_u)

            phi_test_matrix = np.array(phi_test_vector).reshape(1, -1)

            y_pred = np.dot(phi_test_matrix, theta_matrix.T) + model.intercept_

            y_test.append(float(y_pred[0]))
    
    return y_test[-400:]