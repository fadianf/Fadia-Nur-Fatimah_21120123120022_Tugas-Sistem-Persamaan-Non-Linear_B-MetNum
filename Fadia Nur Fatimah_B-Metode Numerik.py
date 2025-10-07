"""
Solver Sistem Persamaan Nonlinear
Nama: Fadia Nur Fatimah
NIM: 21120123120022
NIMx = 22 mod 4 = 2
Kombinasi: g1B dan g2A

Sistem Persamaan:
f1(x,y) = x² + xy - 10 = 0
f2(x,y) = y + 3xy² - 57 = 0

Solusi sejati: x = 2, y = 3
"""

import math

# ==================== FUNGSI DASAR ====================
def f1(x, y):
    return x**2 + x*y - 10

def f2(x, y):
    return y + 3*x*y**2 - 57

# ==================== FUNGSI ITERASI ====================
def g1B(x, y):
    val = 10 - x*y
    if val <= 0:
        raise ValueError("Nilai negatif di akar pada g1B.")
    return math.sqrt(val)

def g2A(x, y):
    val = (57 - y) / (3*x)
    if val <= 0:
        raise ValueError("Nilai negatif di akar pada g2A.")
    return math.sqrt(val)

# ==================== TURUNAN UNTUK NEWTON-RAPHSON ====================
def df1_dx(x, y): return 2*x + y
def df1_dy(x, y): return x
def df2_dx(x, y): return 3*y**2
def df2_dy(x, y): return 1 + 6*x*y

# ==================== METODE ITERASI TITIK TETAP - JACOBI ====================
def iterasi_jacobi(g1, g2, x0, y0, epsilon=1e-6, max_iter=100):
    print("\n" + "="*70)
    print("METODE ITERASI TITIK TETAP - JACOBI")
    print("="*70)
    print(f"{'Iter':<5} {'x':>10} {'y':>10} {'Δx':>12} {'Δy':>12}")
    print("-"*70)

    x, y = x0, y0
    print(f"{0:<5} {x:>10.6f} {y:>10.6f} {0:>12.6f} {0:>12.6f}")

    for i in range(1, max_iter + 1):
        try:
            x_new = g1(x, y)
            y_new = g2(x, y)
        except Exception as e:
            print(f"\nERROR iterasi {i}: {e}")
            return None, None, False, i

        dx, dy = abs(x_new - x), abs(y_new - y)
        print(f"{i:<5} {x_new:>10.6f} {y_new:>10.6f} {dx:>12.6f} {dy:>12.6f}")

        if dx < epsilon and dy < epsilon:
            print(f"\nKONVERGEN pada iterasi {i}")
            print(f"x = {x_new:.6f}, y = {y_new:.6f}")
            print(f"f1 = {f1(x_new, y_new):.6e}, f2 = {f2(x_new, y_new):.6e}")
            return x_new, y_new, True, i
        x, y = x_new, y_new
    print("\nTidak konvergen setelah", max_iter, "iterasi.")
    return x, y, False, max_iter

# ==================== METODE ITERASI TITIK TETAP - SEIDEL ====================
def iterasi_seidel(g1, g2, x0, y0, epsilon=1e-6, max_iter=100):
    print("\n" + "="*70)
    print("METODE ITERASI TITIK TETAP - SEIDEL (GAUSS-SEIDEL)")
    print("="*70)
    print(f"{'Iter':<5} {'x':>10} {'y':>10} {'Δx':>12} {'Δy':>12}")
    print("-"*70)

    x, y = x0, y0
    print(f"{0:<5} {x:>10.6f} {y:>10.6f} {0:>12.6f} {0:>12.6f}")

    for i in range(1, max_iter + 1):
        try:
            x_new = g1(x, y)
            y_new = g2(x_new, y)
        except Exception as e:
            print(f"\nERROR iterasi {i}: {e}")
            return None, None, False, i

        dx, dy = abs(x_new - x), abs(y_new - y)
        print(f"{i:<5} {x_new:>10.6f} {y_new:>10.6f} {dx:>12.6f} {dy:>12.6f}")

        if dx < epsilon and dy < epsilon:
            print(f"\nKONVERGEN pada iterasi {i}")
            print(f"x = {x_new:.6f}, y = {y_new:.6f}")
            print(f"f1 = {f1(x_new, y_new):.6e}, f2 = {f2(x_new, y_new):.6e}")
            return x_new, y_new, True, i
        x, y = x_new, y_new
    print("\nTidak konvergen setelah", max_iter, "iterasi.")
    return x, y, False, max_iter

# ==================== METODE NEWTON-RAPHSON ====================
def newton_raphson(x0, y0, epsilon=1e-6, max_iter=100):
    print("\n" + "="*70)
    print("METODE NEWTON-RAPHSON")
    print("="*70)
    print(f"{'Iter':<5} {'x':>10} {'y':>10} {'Δx':>12} {'Δy':>12}")
    print("-"*70)

    x, y = x0, y0
    print(f"{0:<5} {x:>10.6f} {y:>10.6f} {0:>12.6f} {0:>12.6f}")

    for i in range(1, max_iter + 1):
        f1v, f2v = f1(x, y), f2(x, y)
        J = df1_dx(x, y)*df2_dy(x, y) - df1_dy(x, y)*df2_dx(x, y)
        if abs(J) < 1e-12:
            print(f"\nDeterminan mendekati nol pada iterasi {i}")
            return None, None, False, i

        x_new = x - (f1v*df2_dy(x, y) - f2v*df1_dy(x, y)) / J
        y_new = y - (f2v*df1_dx(x, y) - f1v*df2_dx(x, y)) / J
        dx, dy = abs(x_new - x), abs(y_new - y)
        print(f"{i:<5} {x_new:>10.6f} {y_new:>10.6f} {dx:>12.6f} {dy:>12.6f}")

        if dx < epsilon and dy < epsilon:
            print(f"\nKONVERGEN pada iterasi {i}")
            print(f"x = {x_new:.6f}, y = {y_new:.6f}")
            print(f"f1 = {f1(x_new, y_new):.6e}, f2 = {f2(x_new, y_new):.6e}")
            return x_new, y_new, True, i
        x, y = x_new, y_new
    print("\nTidak konvergen setelah", max_iter, "iterasi.")
    return x, y, False, max_iter

# ==================== METODE SECANT ====================
def secant_method(x0, y0, x1, y1, epsilon=1e-6, max_iter=100):
    print("\n" + "="*70)
    print("METODE SECANT")
    print("="*70)
    print(f"{'Iter':<5} {'x':>10} {'y':>10} {'Δx':>12} {'Δy':>12}")
    print("-"*70)

    x_prev, y_prev = x0, y0
    x, y = x1, y1
    print(f"{0:<5} {x_prev:>10.6f} {y_prev:>10.6f} {0:>12.6f} {0:>12.6f}")
    print(f"{1:<5} {x:>10.6f} {y:>10.6f} {abs(x-x_prev):>12.6f} {abs(y-y_prev):>12.6f}")

    for i in range(2, max_iter + 1):
        f1v, f2v = f1(x, y), f2(x, y)
        f1p, f2p = f1(x_prev, y_prev), f2(x_prev, y_prev)
        dx, dy = x - x_prev, y - y_prev

        du_dx = (f1v - f1p) / dx
        du_dy = (f1(x, y + 0.0001) - f1v) / 0.0001
        dv_dx = (f2v - f2p) / dx
        dv_dy = (f2(x, y + 0.0001) - f2v) / 0.0001
        det = du_dx * dv_dy - du_dy * dv_dx

        if abs(det) < 1e-10:
            print(f"\nDeterminan mendekati nol pada iterasi {i}")
            return None, None, False, i

        x_new = x - (f1v * dv_dy - f2v * du_dy) / det
        y_new = y + (f1v * dv_dx - f2v * du_dx) / det
        dx, dy = abs(x_new - x), abs(y_new - y)
        print(f"{i:<5} {x_new:>10.6f} {y_new:>10.6f} {dx:>12.6f} {dy:>12.6f}")

        if dx < epsilon and dy < epsilon:
            print(f"\nKONVERGEN pada iterasi {i}")
            print(f"x = {x_new:.6f}, y = {y_new:.6f}")
            print(f"f1 = {f1(x_new, y_new):.6e}, f2 = {f2(x_new, y_new):.6e}")
            return x_new, y_new, True, i
        x_prev, y_prev, x, y = x, y, x_new, y_new
    print("\nTidak konvergen setelah", max_iter, "iterasi.")
    return x, y, False, max_iter

# ==================== MAIN ====================
def main():
    print("="*70)
    print("PENYELESAIAN SISTEM PERSAMAAN NONLINEAR")
    print("="*70)
    print("Nama: Fadia Nur Fatimah")
    print("NIM: 21120123120022")
    print("NIMx = 22 mod 4 = 2")
    print("Kombinasi Fungsi Iterasi: g1B dan g2A")
    print("="*70)

    x0, y0 = 1.5, 3.5
    eps, max_iter = 1e-6, 100

    results = []
    print("\n1️⃣  ITERASI TITIK TETAP - JACOBI (g1B,g2A)")
    x, y, conv, it = iterasi_jacobi(g1B, g2A, x0, y0, eps, max_iter)
    results.append(("Jacobi", x, y, conv, it))

    print("\n2️⃣  ITERASI TITIK TETAP - SEIDEL (g1B,g2A)")
    x, y, conv, it = iterasi_seidel(g1B, g2A, x0, y0, eps, max_iter)
    results.append(("Seidel", x, y, conv, it))

    print("\n3️⃣  METODE NEWTON-RAPHSON")
    x, y, conv, it = newton_raphson(x0, y0, eps, max_iter)
    results.append(("Newton-Raphson", x, y, conv, it))

    print("\n4️⃣  METODE SECANT")
    x1, y1 = 1.4, 3.6
    x, y, conv, it = secant_method(x0, y0, x1, y1, eps, max_iter)
    results.append(("Secant", x, y, conv, it))

    print("\n" + "="*70)
    print("RINGKASAN HASIL")
    print("="*70)
    print(f"{'Metode':<20}{'x':>10}{'y':>10}{'Status':>15}{'Iterasi':>10}")
    print("-"*70)
    for m, xv, yv, c, it in results:
        status = "Konvergen" if c else "Divergen"
        x_str = f"{xv:.6f}" if xv else "N/A"
        y_str = f"{yv:.6f}" if yv else "N/A"
        print(f"{m:<20}{x_str:>10}{y_str:>10}{status:>15}{it:>10}")
    print("="*70)

if __name__ == "__main__":
    main()
