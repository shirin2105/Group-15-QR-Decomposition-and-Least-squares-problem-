import numpy as np

def back_substitution(R, d):
    """
    Giải hệ phương trình tam giác trên Rx = d.
    """
    n = R.shape[1]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) < 1e-15:
            x[i] = 0
        else:
            sum_val = np.dot(R[i, i+1:], x[i+1:])
            x[i] = (d[i] - sum_val) / R[i, i]
    return x
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.copy().astype(float)
    
    for i in range(n):
        # Tính chuẩn của vector hiện tại (độ dài cạnh huyền)
        R[i, i] = np.linalg.norm(V[:, i])
        
        if R[i, i] > 1e-15:
            # Chuẩn hóa vector để lấy hướng (vector đơn vị q)
            Q[:, i] = V[:, i] / R[i, i]
            
            # Khử hình chiếu của Q[:, i] trên các cột còn lại
            for j in range(i + 1, n):
                R[i, j] = np.dot(Q[:, i], V[:, j])
                V[:, j] -= R[i, j] * Q[:, i]
                
    # Rx = Q.T * b
    d = np.dot(Q.T, b)
    return back_substitution(R, d)

if __name__ == "__main__":
    A_sample = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    b_sample = np.array([7, 8, 9], dtype=float)
    
    x_sol = gram_schmidt_qr_solver(A_sample, b_sample)
    
    print("--- Kết quả giải bằng Gram-Schmidt QR ---")
    print(f"Nghiệm x: {x_sol}")
    
    x_np, _, _, _ = np.linalg.lstsq(A_sample, b_sample, rcond=None)

    print(f"Nghiệm từ numpy.linalg.lstsq: {x_np}")
