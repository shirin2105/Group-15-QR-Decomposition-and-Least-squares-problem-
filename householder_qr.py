import numpy as np

def back_substitution(R, d):
    """
    Giải hệ phương trình tam giác trên Rx = d.
    R: Ma trận tam giác trên n x n.
    d: Vector n x 1.
    """
    n = R.shape[1]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) < 1e-15:  # Tránh chia cho 0 nếu ma trận suy biến
            x[i] = 0
        else:
            sum_val = np.dot(R[i, i+1:], x[i+1:])
            x[i] = (d[i] - sum_val) / R[i, i]
    return x

def householder_qr_solver(A, b):
    m, n = A.shape
    R = A.copy().astype(float)
    Q = np.eye(m)
    
    for k in range(n):
        # Lấy cột thứ k từ hàng thứ k trở xuống
        x = R[k:, k]
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-15: continue
            
        v = x.copy()
        # Chọn dấu để tránh triệt tiêu số học
        v[0] += np.sign(x[0]) * norm_x
        v /= np.linalg.norm(v)
        
        # Cập nhật R bằng cách áp dụng ma trận Householder H: R = H * R
        # H = I - 2vv^T. H*A = A - 2v(v^T * A)
        R[k:, k:] -= 2 * np.outer(v, np.dot(v, R[k:, k:]))
        
        # Cập nhật Q: Q = Q * H
        Q[:, k:] -= 2 * np.outer(np.dot(Q[:, k:], v), v)
        
    #Ax = b <=> QRx = b <=> Rx = Q.T * b
    d = np.dot(Q.T, b)
    
    # Chỉ lấy phần ma trận vuông n x n phía trên của R và n phần tử đầu của d
    return back_substitution(R[:n, :n], d[:n])

if __name__ == "__main__":
    A_sample = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    b_sample = np.array([7, 8, 9], dtype=float)
    
    x_sol = householder_qr_solver(A_sample, b_sample)
    
    print("--- Kết quả giải bằng Householder QR ---")
    print(f"Nghiệm x: {x_sol}")
    
    # Kiểm tra lại bằng thư viện numpy
    x_np, _, _, _ = np.linalg.lstsq(A_sample, b_sample, rcond=None)

    print(f"Nghiệm từ numpy.linalg.lstsq: {x_np}")
