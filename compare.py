import numpy as np
import matplotlib.pyplot as plt
import time


def back_substitution(R, d):
    """Giải hệ phương trình tam giác trên Rx = d"""
    n = R.shape[1]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) < 1e-15: # Tránh chia cho 0
            x[i] = 0
        else:
            sum_val = np.dot(R[i, i+1:], x[i+1:])
            x[i] = (d[i] - sum_val) / R[i, i]
    return x

def gaussian_elimination(A, b):
    n = len(b)
    M = np.hstack((A.astype(float), b.reshape(-1, 1).astype(float)))
    
    for i in range(n):
        # Tìm dòng có phần tử pivot lớn nhất
        max_row = np.argmax(np.abs(M[i:, i])) + i
        M[i], M[max_row] = M[max_row].copy(), M[i].copy()
        
        if abs(M[i, i]) < 1e-18: continue
            
        for j in range(i + 1, n):
            ratio = M[j, i] / M[i, i]
            M[j, i:] -= ratio * M[i, i:]
            
    return back_substitution(M[:, :-1], M[:, -1])

def householder_qr_solver(A, b):
    m, n = A.shape
    R = A.copy().astype(float)
    Q = np.eye(m)
    
    for k in range(n):
        x = R[k:, k]
        norm_x = np.linalg.norm(x)
        if norm_x < 1e-15: continue
            
        v = x.copy()
        v[0] += np.sign(x[0]) * norm_x
        v /= np.linalg.norm(v)
        
        # R = H*R
        R[k:, k:] -= 2 * np.outer(v, np.dot(v, R[k:, k:]))
        # Q = Q*H
        Q[:, k:] -= 2 * np.outer(np.dot(Q[:, k:], v), v)
        
    d = np.dot(Q.T, b)
    return back_substitution(R[:n, :n], d[:n])

def gram_schmidt_qr_solver(A, b):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = A.copy().astype(float)
    
    for i in range(n):
        R[i, i] = np.linalg.norm(V[:, i])
        if R[i, i] > 1e-15:
            Q[:, i] = V[:, i] / R[i, i]
            for j in range(i + 1, n):
                R[i, j] = np.dot(Q[:, i], V[:, j])
                V[:, j] -= R[i, j] * Q[:, i]
                
    d = np.dot(Q.T, b)
    return back_substitution(R, d)

def normal_equation_solver(A, b):
    ATA = A.T @ A
    ATb = A.T @ b
    return gaussian_elimination(ATA, ATb)


# HÀM HỖ TRỢ SINH DỮ LIỆU
def generate_ill_conditioned_matrix(m, n, cond_num):
    """Tạo ma trận có số điều kiện cụ thể"""
    X = np.random.randn(m, n)
    U, _ = np.linalg.qr(X) # Dùng QR thư viện chỉ để tạo ma trận trực giao mẫu
    V, _ = np.linalg.qr(np.random.randn(n, n))
    s = np.logspace(0, np.log10(1/cond_num), n)
    A = U @ np.diag(s) @ V.T
    return A


#thực nghiệm và đánh giá
def run_stability_experiment():
    print("--- 1. Đánh giá ĐỘ ỔN ĐỊNH (Stability) ---")
    m, n = 100, 20
    cond_numbers = np.logspace(1, 16, 15) 
    
    err_hh, err_gs, err_ne = [], [], []
    
    for cond in cond_numbers:
        A = generate_ill_conditioned_matrix(m, n, cond)
        x_true = np.random.randn(n)
        b = A @ x_true 
        
        # 1. Householder
        x_hh = householder_qr_solver(A, b)
        err_hh.append(np.linalg.norm(x_hh - x_true) / np.linalg.norm(x_true))
        
        # 2. Gram-Schmidt
        x_gs = gram_schmidt_qr_solver(A, b)
        err_gs.append(np.linalg.norm(x_gs - x_true) / np.linalg.norm(x_true))
        
        # 3. Normal Equation
        x_ne = normal_equation_solver(A, b)
        err_ne.append(np.linalg.norm(x_ne - x_true) / np.linalg.norm(x_true))

    plt.figure(figsize=(10, 6))
    plt.loglog(cond_numbers, err_hh, 'b-o', label='Householder QR')
    plt.loglog(cond_numbers, err_gs, 'g-^', label='Gram-Schmidt QR')
    plt.loglog(cond_numbers, err_ne, 'r--s', label='Normal Equation')
    
    plt.axhline(y=np.finfo(float).eps, color='k', linestyle=':', label='Machine Epsilon')
    plt.title(f'So sánh độ chính xác (Matrix {m}x{n})')
    plt.xlabel('Condition Number (cond(A))')
    plt.ylabel('Sai số tương đối')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()

def run_time_experiment():
    print("\n--- 2. Đánh giá THỜI GIAN (Speed) ---")
    sizes = [100, 200, 400, 600, 800] 
    t_hh, t_gs, t_ne = [], [], []
    
    for size in sizes:
        m, n = size * 2, size
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        
        # Thời gian Householder
        s = time.time(); householder_qr_solver(A, b); t_hh.append(time.time() - s)
        # Thời gian Gram-Schmidt
        s = time.time(); gram_schmidt_qr_solver(A, b); t_gs.append(time.time() - s)
        # Thời gian Normal Equation
        s = time.time(); normal_equation_solver(A, b); t_ne.append(time.time() - s)
        
        print(f"Size {size}: HH={t_hh[-1]:.4f}s, GS={t_gs[-1]:.4f}s, NE={t_ne[-1]:.4f}s")
        
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, t_hh, 'b-o', label='Householder QR')
    plt.plot(sizes, t_gs, 'g-^', label='Gram-Schmidt QR')
    plt.plot(sizes, t_ne, 'r--s', label='Normal Equation')
    
    plt.title('So sánh thời gian thực thi (Tự cài đặt thuần Python)')
    plt.xlabel('Kích thước n (m=2n)')
    plt.ylabel('Thời gian (giây)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    run_stability_experiment()

    run_time_experiment()
