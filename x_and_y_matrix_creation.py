def X_matrix_creation(n,m,d,train_u,train_y):

    p = max(n, m + d)

    N = len(train_u)

    X_matrix = []

    def create_phi_matrix(k, n, m):

        y_theta = train_y[k-n:k]  
        y_theta = y_theta[::-1]  

        u_theta = train_u[k-d-m:k-d+1] 
        u_theta = u_theta[::-1] 
    
        phi = np.concatenate((y_theta, u_theta))

        return phi

    for k in range(p, N):  
        phi_matrix = create_phi_matrix(k, n, m)
        X_matrix.append(phi_matrix) 

    X_matrix = np.array(X_matrix)

    return X_matrix

def Y_matrix_creation(n,m,d,train_y):

    p = max(n,m+d)
    N = len(train_u)
    
    Y_matrix = []

    for k in range(p,N):
        y_p = np.array(train_y[k]) 
    
        Y_matrix = np.append(Y_matrix,y_p) 

    Y_matrix = np.array(Y_matrix).reshape(-1, 1) 

    return Y_matrix
    