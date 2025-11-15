from typing import Tuple, List
import sys
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager as fm
import numpy as np


# ============================================================
# 日本語フォントのクロスプラットフォーム設定（警告対策）
# ------------------------------------------------------------
# 目的:
#   - macOS で Meiryo が存在しないため発生していた
#     "findfont: Generic family 'sans-serif' not found..." 警告を解消する。
#   - OSごとに妥当な日本語フォント候補のリストを用意し、
#     利用可能な最初のフォントを選択して rcParams に設定する。
#
# ポイント:
#   - rcParams["font.sans-serif"] に複数候補を渡しておけば、
#     見つかった最初のフォントが使われるため警告が出にくい。
#   - "axes.unicode_minus=False" を設定しておくと、マイナス記号の文字化けを防げる。
# ============================================================
def _setup_japanese_font() -> str:
    # OS別に候補フォントを用意
    if sys.platform == "darwin":  # macOS
        candidates = [
            "Hiragino Sans",
            "Hiragino Kaku Gothic ProN",
            "Noto Sans CJK JP",
            "Yu Gothic",
            "Meiryo",
            "IPAexGothic",
            "DejaVu Sans",
        ]
    elif sys.platform.startswith("win"):  # Windows
        candidates = [
            "Meiryo",
            "Yu Gothic",
            "Noto Sans CJK JP",
            "IPAexGothic",
            "MS Gothic",
            "DejaVu Sans",
        ]
    else:  # Linux 等
        candidates = [
            "Noto Sans CJK JP",
            "IPAPGothic",
            "IPAexGothic",
            "VL Gothic",
            "TakaoGothic",
            "Yu Gothic",
            "Meiryo",
            "DejaVu Sans",
        ]

    # 利用可能なフォント名を収集
    available = {f.name for f in fm.fontManager.ttflist}

    # 最初に見つかった候補を採用
    picked = None
    for name in candidates:
        if name in available:
            picked = name
            break
    if picked is None:
        picked = "DejaVu Sans"  # 最低限の英数字フォールバック

    rcParams["font.family"] = "sans-serif"
    # 最初の要素に picked、その後に候補を並べる（よりよいフォールバック）
    rcParams["font.sans-serif"] = [picked] + [n for n in candidates if n != picked]
    rcParams["axes.unicode_minus"] = False
    return picked


# 実行時に一度だけ設定
_selected_font = _setup_japanese_font()


"""
本スクリプトは、単位正方形領域 Ω = [0, Lx] × [0, Ly]（ここでは Lx = Ly = H*M）上の
2 次元ラプラス方程式
    ∇^2 u = u_xx + u_yy = 0
に対して、有限差分法（FDM: 5 点ラプラシアン）とガウス–ザイデル法による反復で
ディリクレ境界値問題を解きます。

離散化（等間隔格子, 格子幅 H）:
  格子点 (x_i, y_j) = (i H, j H),   i = 0..M, j = 0..N  （節点数は (M+1)×(N+1)）
  5 点近似:  (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4 u_{i,j}) / H^2 = 0
  ⇒ 内点についての更新式（厳密解が調和関数なので右辺 0）:
     u_{i,j} = (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1}) / 4

反復法:
  ガウス–ザイデル法（SOR の ω=1 特例）を用い、最新の近傍値で逐次更新します。
  収束性: 本問題の離散化で得られる係数行列は対称正定値（SPD）かつ M 行列の性質を持ち、
  ガウス–ザイデルは収束します。収束速度は格子幅に依存し、概ね 1 - O(h^2) 程度の収束係数。

境界条件:
  下辺 y=0, 左辺 x=0, 右辺 x=Lx は u=0、上辺 y=Ly は u(x, Ly)= f(x) = 4x(1-x)
  （x は物理座標、ここでは Lx=1 を想定した関数形。H*M=1 となる設定例ではそのまま一致）。

誤差と停止条件:
  現在解 U と前回解 UF の差の L1 型相対量（∑|Δ| / ∑|U|）が許容値以下で停止。
  ※これは「反復差」基準であり、厳密な残差 ||A U - b|| 基準ではない点に注意。
  精度は空間 O(H^2)（5 点ラプラシアン）を期待。
"""


def _set_initial_condition(
    *,
    array_2d: List[List[float]],
    grid_counts_x: int,
    grid_counts_y: int,
) -> None:
    """
    初期条件（内部点の初期推定値）の設定。
    ここでは内部全点を 1.0e-4 の小値で初期化します（0 でもよいが、相対誤差基準での
    0 除算回避の助けになります）。境界点は後で境界条件で上書きされます。

    引数:
        array_2d: U 値の 2 次元配列（サイズ (M+1)×(N+1)）
        grid_counts_x: M（x 方向分割数, 節点は 0..M）
        grid_counts_y: N（y 方向分割数, 節点は 0..N）
    """
    for j in range(1, grid_counts_y):
        for i in range(1, grid_counts_x):
            array_2d[i][j] = 0.0001


def _set_boundary_condition(
    *,
    array_2d: List[List[float]],
    grid_counts_x: int,
    grid_counts_y: int,
    grid_space: float,
) -> None:
    """
    ディリクレ境界条件の設定。
    本例では
      ・下辺 y=0, 左辺 x=0, 右辺 x=H*M: u=0
      ・上辺 y=H*N: u(x, Ly)=f(x)=4x(1-x)
    を課します。ここで x は物理座標、x = i * H。

    引数:
        array_2d: U の 2 次元配列
        grid_counts_x: M
        grid_counts_y: N
        grid_space: H
    """
    # y=0（下辺）
    for i in range(grid_counts_x + 1):
        array_2d[i][0] = 0.0

    # y=Ly（上辺, j=N）
    for i in range(grid_counts_x + 1):
        x_coordinate = grid_space * i
        # 連続境界関数 f(x)=4x(1-x) を節点に代入
        array_2d[i][grid_counts_y] = 4.0 * x_coordinate * (1.0 - x_coordinate)

    # x=0（左辺, i=0）
    for j in range(1, grid_counts_y):
        array_2d[0][j] = 0.0

    # x=Lx（右辺, i=M）
    for j in range(1, grid_counts_y):
        array_2d[grid_counts_x][j] = 0.0


def _is_converged(
    *, U: List[List[float]], UF: List[List[float]], M: int, N: int
) -> bool:
    """
    収束判定（反復差に基づく相対誤差）。
    error_sum = Σ|U^{k+1}_{i,j} - U^{k}_{i,j}|（内部点）
    value_sum = Σ|U^{k+1}_{i,j}|（内部点）
    relative_error = error_sum / value_sum
    が閾値以下なら停止とします。value_sum≈0 の初期段階では早期停止を避けるための
    防御的な処理を入れます。

    注: これは残差 ||A U - b|| ではないため、厳密な PDE 残差停止条件よりは緩い可能性があります。

    引数:
        U: 現在の解
        UF: 前回の解
        M, N: 分割数
    """
    ERROR_CONSTANT: float = 1.0e-4
    error_sum: float = 0.0
    value_sum: float = 0.0

    for j in range(1, N):
        for i in range(1, M):
            value_sum += abs(U[i][j])
            error_sum += abs(U[i][j] - UF[i][j])

    # ゼロ除算の回避と、初期段階の不安定な相対値の抑制
    if value_sum < 1.0e-10:
        return False  # 初期は収束とみなさないほうが安全

    relative_error = error_sum / value_sum
    return relative_error <= ERROR_CONSTANT


def calculate_equation(
    *, M: int, N: int, H: float, MK: int
) -> Tuple[List[List[float]], int]:
    """
    ガウス–ザイデル反復でラプラス方程式の離散系を解く。

    5 点ラプラシアンの内部点での離散方程式は
        -4 u_{i,j} + u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} = 0
    であり、更新式（Gauss–Seidel）は
        u_{i,j}^{(k+1)} = (u_{i+1,j}^{(k)}(or k+1) + u_{i-1,j}^{(k+1)}
                           + u_{i,j+1}^{(k)} + u_{i,j-1}^{(k+1)}) / 4
    の形になります（左から右、下から上の走査順により新しい値を随時使用）。

    引数:
        M, N: 分割数（節点は 0..M, 0..N）
        H: 格子幅
        MK: 最大反復回数

    戻り値:
        (U, calc_count): 収束時（または打ち切り時）の解行列と総反復回数
    """
    assert (
        M >= 2 and N >= 2
    ), "内部点が存在するように M, N は 2 以上である必要があります。"
    assert H > 0.0, "格子幅 H は正である必要があります。"

    # U: 現在解, UF: 収束判定用の前回解（サイズ (M+1)×(N+1)）
    U: List[List[float]] = [[0.0 for _ in range(N + 1)] for _ in range(M + 1)]
    UF: List[List[float]] = [[0.0 for _ in range(N + 1)] for _ in range(M + 1)]

    # 初期条件（内部点）と境界条件の設定
    _set_initial_condition(array_2d=U, grid_counts_x=M, grid_counts_y=N)
    _set_boundary_condition(array_2d=U, grid_counts_x=M, grid_counts_y=N, grid_space=H)

    calc_count: int = 0

    for _ in range(MK):
        # 収束判定用に UF ← U をコピー（深い二重ループコピー）
        for j in range(N + 1):
            for i in range(M + 1):
                UF[i][j] = U[i][j]

        # 内部点をガウス–ザイデル更新
        # 5 点平均（ラプラシアン=0）: 二次精度（O(H^2)）
        for j in range(1, N):
            for i in range(1, M):
                # 新しい値（i-1, j）、（i, j-1）はすでに更新済みを利用
                U[i][j] = (U[i + 1][j] + U[i - 1][j] + U[i][j + 1] + U[i][j - 1]) / 4.0

        calc_count += 1

        # 反復差に基づく収束判定
        if _is_converged(U=U, UF=UF, M=M, N=N):
            print("収束しました")
            break
    else:
        # 上の for を正常終了できなかった場合（break されなかった）
        print(f"警告: 最大反復回数({MK})に達しました。収束していない可能性があります。")

    return U, calc_count


def color_plot(
    *,
    array_2d: List[List[float]],
    grid_counts: int,
    grid_space: float,
) -> None:
    """
    2 次元配列を疑似カラーで可視化（imshow）。
    extent を [-H/2, L+H/2] に取るのは、セル中心と座標系の対応を視覚上合わせる意図です。

    引数:
        array_2d: U の 2 次元配列（(M+1)×(N+1)）
        grid_counts: ここでは M=N を想定して同一値を渡します
        grid_space: H
    """
    # 表示範囲（セル中心が [0, L] に並ぶように少し拡げる）
    min_x_y = 0.0 - grid_space / 2.0
    max_x_y = grid_space * grid_counts + grid_space / 2.0
    extent = (min_x_y, max_x_y, min_x_y, max_x_y)

    # imshow は [行, 列] = [y, x] を前提とするので転置
    array_2d_transposed = np.array(array_2d).T

    plt.figure(figsize=(8, 6))
    plt.imshow(
        array_2d_transposed,
        cmap="viridis",
        interpolation="none",
        aspect="auto",
        origin="lower",
        extent=extent,
    )
    plt.colorbar(label="Temperature (U)")
    plt.title("2D Temperature Distribution (Laplace Equation)")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.xticks(np.arange(0.0, grid_space * (grid_counts + 1), grid_space * 2))
    plt.yticks(np.arange(0.0, grid_space * (grid_counts + 1), grid_space * 2))

    plt.savefig("./2d_color_plot.png", format="png", dpi=300)
    plt.close()


def output_result_file(
    array_2d: List[List[float]],
    grid_counts_x: int,
    grid_counts_y: int,
    grid_space: float,
    calc_count: int,
) -> None:
    """
    計算パラメータと最終解行列 U をテキスト出力。

    引数:
        array_2d: U
        grid_counts_x: M
        grid_counts_y: N
        grid_space: H
        calc_count: 反復回数
    """
    output_filename = "./calculated_result.txt"
    with open(output_filename, "w", encoding="utf-8") as file:
        # ヘッダ
        file.write(
            "# Calculated result of Laplace equation using Gauss-Seidel method.\n\n"
        )

        # 計算パラメータ
        file.write("# Calculation Parameters:\n")
        file.write(f"#   grid_counts_x: {grid_counts_x}\n")
        file.write(f"#   grid_counts_y: {grid_counts_y}\n")
        file.write(f"#   grid_space (H): {grid_space}\n")
        file.write(f"#   calculation_count: {calc_count}\n\n")

        # 解行列 U
        file.write("# Calculated Matrix U (Temperature Distribution):\n")
        for row in array_2d:
            line = " ".join(map(str, row))
            file.write(line + "\n")

    print(f"計算結果を {output_filename} に保存しました。")


if __name__ == "__main__":
    """
    実行フロー:
      1) パラメータ設定（ここでは Lx=Ly=H*M とみなす）
      2) ガウス–ザイデルで差分方程式を反復（calculate_equation）
      3) 可視化（color_plot）
      4) テキスト出力（output_result_file）

    解析的参照解（任意）:
      本境界値問題はフーリエ正弦級数で
        u(x,y) = Σ_{n=1}^∞ b_n * sinh(nπ y / Lx) / sinh(nπ Ly / Lx) * sin(nπ x / Lx)
      の形で与えられます（上辺値 f(x)=4x(1-x) の正弦展開係数 b_n を用いる）。
      これにより離散解の検証が可能です（本コードでは省略）。
    """
    # --- パラメータ設定 ---
    grid_counts_x: int = 10  # M: x 分割数（節点は 0..M）
    grid_counts_y: int = 10  # N: y 分割数（節点は 0..N）
    grid_space: float = 0.1  # H: 格子幅（例: H*M = 1.0）
    max_iterations: int = 1000  # MK: 最大反復回数

    print(f"選択フォント: { _selected_font }")
    print("計算を開始します...")

    # 差分方程式の計算を実行
    final_solution, iterations = calculate_equation(
        M=grid_counts_x,
        N=grid_counts_y,
        H=grid_space,
        MK=max_iterations,
    )

    print("結果を可視化・保存します...")

    # 可視化（正方格子を仮定して grid_counts_x を渡す）
    color_plot(
        array_2d=final_solution,
        grid_counts=grid_counts_x,
        grid_space=grid_space,
    )

    # 結果のファイル出力
    output_result_file(
        array_2d=final_solution,
        grid_counts_x=grid_counts_x,
        grid_counts_y=grid_counts_y,
        grid_space=grid_space,
        calc_count=iterations,
    )

    print("処理が完了しました。")
