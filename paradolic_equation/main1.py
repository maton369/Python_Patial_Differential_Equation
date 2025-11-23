# ================================================================
# 1D 熱伝導方程式 u_t = χ u_xx の Crank–Nicolson（半陰解法）+ ガウス–ザイデル反復ソルバ
# ---------------------------------------------------------------
# ● PDEと格子
#   - 連続系:   ∂u/∂t = χ ∂²u/∂x²  （0≤x≤L, t≥0）
#   - 格子点:   x_i = i Δx  (i=0..M),  t_j = j Δt  (j=0..N)
#
# ● Crank–Nicolson(CN) の標準離散化（内部点 i=1..M-1）:
#
#   (u_i^{j+1}-u_i^j)/Δt = (χ/2) * [ (u_{i+1}^{j+1} - 2u_i^{j+1} + u_{i-1}^{j+1})/Δx²
#                                   + (u_{i+1}^{j}   - 2u_i^{j}   + u_{i-1}^{j})  /Δx² ]
#
#   R := χ Δt / Δx² とおくと、
#
#   (1+R) u_i^{j+1} - (R/2)(u_{i+1}^{j+1}+u_{i-1}^{j+1})
#     = (1-R) u_i^{j} + (R/2)(u_{i+1}^{j} + u_{i-1}^{j})
#
#   すなわち A u^{j+1} = b^j の三重対角(対称/対角優位)連立一次方程式が各時刻で現れる。
#
# ● 本コードのガウス–ザイデル更新式との対応
#   上式の両辺を (-2/R) 倍すると
#
#   -2(1+1/R) u_i^{j+1} + u_{i+1}^{j+1} + u_{i-1}^{j+1}
#     = -(u_{i+1}^{j} - 2u_i^{j} + u_{i-1}^{j}) - (2/R) u_i^{j}
#
#   と書ける。ここで
#     BS := -2(1+1/R),   BL := -(Δ_x² u^j)_i - (2/R) u_i^j
#   と定義すれば、ガウス–ザイデル反復は
#     u_i^{j+1,(k+1)} = ( BL - u_{i+1}^{j+1,(k)} - u_{i-1}^{j+1,(k+1)} ) / BS
#   となる。本実装の BS, BL はこの形をそのままコーディングしたものである。
#
# ● 収束と安定性
#   - CN は A-安定（時間刻みに対する制約が緩い）ため、陽解法FTCSの R≤1/2 の制約を受けない。
#   - ただし各時刻での反復ソルバ（ここではGS）には停止判定が必要。本コードは
#     「相対L1差分 ≤ TOLERANCE」を収束基準として用いる。
#
# ● 可視化・出力
#   - 各時刻の温度プロファイルを GIF にし、結果行列（空間×時間）をテキストで保存する。
# ================================================================

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import ArtistAnimation

# --- 定数定義 ---
# 生成するアニメーションGIFファイルのパス
ANIMATION_GIF_PATH = "./animation.gif"
# 計算結果を出力するテキストファイルのパス
RESULT_FILE_PATH = "./calculated_result.txt"
# ガウス-ザイデル法の収束判定許容誤差（相対L1差分）
TOLERANCE = 1e-6


def set_initial_condition(u: np.ndarray) -> None:
    """
    初期条件 (時刻 t=0) を設定します。
    棒の内部 (境界を除く) の初期温度を 0.0 ℃に設定します。

    Args:
        u (np.ndarray): 温度分布を格納する2次元NumPy配列。
                        形状は (空間格子点数 M+1, 時間ステップ数 N+1)。
                        u[1:-1, 0] の要素が変更されます。

    備考:
        本スクリプトの main では 1次元配列 u_current を直接 0 初期化しており、
        ここは「2次元版の典型的な初期化」を示す参考関数として残している。
    """
    # u[空間点i, 時間点j] で、初期時刻 j=0 の内部点を 0 に。
    u[1:-1, 0] = 0.0


def set_boundary_condition(u: np.ndarray) -> None:
    """
    境界条件 (すべての時刻 t >= 0) を設定します。
    棒の左端 (x=0) を 0.0 ℃、右端 (x=1) を 100.0 ℃に固定します。

    Args:
        u (np.ndarray): 温度分布を格納する2次元NumPy配列。
                        u[0, :] と u[-1, :] の要素が変更されます。

    備考:
        本スクリプトの main では 1次元 u_next に対して境界を直書きしており、
        ここは「2次元表現の雛形」を示す補助関数の位置づけ。
    """
    # 左端 i=0 の温度を 0 ℃、右端 i=M の温度を 100 ℃ に固定（全時刻）。
    u[0, :] = 0.0
    u[-1, :] = 100.0


def is_converged(u_new: np.ndarray, u_old: np.ndarray, M: int) -> bool:
    """
    ガウス-ザイデル法の収束判定を行います。
    現在の反復ステップの解と前回の反復ステップの解の相対誤差を計算し、
    許容誤差 TOLERANCE 以下であれば収束したと判定します。

    基準:
        relative_error = ( Σ_{i=1}^{M-1} |u_new[i]-u_old[i]| )
                         / ( Σ_{i=1}^{M-1} |u_new[i]| + 1e-10 )

    Args:
        u_new (np.ndarray): 現在の反復ステップで計算された温度分布 (1次元配列)。
        u_old (np.ndarray): 前回の反復ステップの温度分布 (1次元配列)。
        M (int): 空間方向の格子分割数。

    Returns:
        bool: 収束していれば True、そうでなければ False。
    """
    # 内部格子点 (i=1..M-1) の差分のL1合計
    error_sum = np.sum(np.abs(u_new[1:M] - u_old[1:M]))
    # 相対化のためのスケール（ゼロ除算防止で微小値を加える）
    value_sum = np.sum(np.abs(u_new[1:M])) + 1e-10
    relative_error = error_sum / value_sum
    return relative_error <= TOLERANCE


def calculate_crank_nicolson_step(
    u_next: np.ndarray,
    u_current: np.ndarray,
    M: int,
    R_chi: float,
    max_iterations: int,
) -> int:
    """
    クランク-ニコルソン法の1時間ステップ分の計算を実行します。
    内部でガウス-ザイデル反復法を用いて連立一次方程式を解きます。

    Args:
        u_next (np.ndarray): 次の時刻 (j+1) の温度分布を格納する1次元配列。
                             この配列の内容が反復計算によって更新される。
        u_current (np.ndarray): 現在の時刻 (j) の既知の温度分布 (1次元配列)。
        M (int): 空間方向の格子分割数。
        R_chi (float): パラメータ R = Δt * χ / (Δx)^2（= χΔt/Δx²）。
                       （上部の理論メモでの R に一致）
        max_iterations (int): ガウス-ザイデル法の最大反復回数。

    Returns:
        int: 収束までにかかったガウス-ザイデル法の反復回数。
             最大反復回数に達した場合は max_iterations を返す。

    理論メモ（式の対応）:
        標準CNの三重対角系 A u^{j+1} = b を (-2/R) 倍した整理で
        BS = -2 * (1 + 1/R),  BL = -(Δ_x² u^j)_i - (2/R) u_i^j
        を用いると、GS更新は
          u_i^{j+1,(k+1)} = (BL - u_{i+1}^{j+1,(k)} - u_{i-1}^{j+1,(k+1)}) / BS
        となる（上部の導出参照）。
    """
    # 反復用ワーク（u_iter が最新推定、u_prev_iter が前回推定）
    u_iter = u_next.copy()
    u_prev_iter = u_next.copy()

    # ガウス-ザイデル反復
    for k in range(max_iterations):
        u_prev_iter[:] = u_iter  # 前回の反復結果を保存

        # 内部点 i=1..M-1 を順次更新（左→右に走査。左側は新値、右側は旧値を使うGS）
        for i in range(1, M):
            # 係数（上の理論メモの記号に対応）
            BS = -2.0 * (1.0 + 1.0 / R_chi)
            # BL は (−ラプラシアン) − (2/R) * u_current
            BL = (
                -(u_current[i + 1] - 2.0 * u_current[i] + u_current[i - 1])
                - (2.0 / R_chi) * u_current[i]
            )

            # ガウス-ザイデル更新
            # 右隣 u_{i+1}^{(k)} は前回反復の値（u_prev_iter）、左隣 u_{i-1}^{(k+1)} は今反復の新値（u_iter）
            u_iter[i] = (BL - u_prev_iter[i + 1] - u_iter[i - 1]) / BS

        # 収束判定（相対L1差分）
        if is_converged(u_iter, u_prev_iter, M):
            u_next[:] = u_iter  # 収束したら次時刻解に確定
            return k + 1  # 実行した反復回数

    # ここに来たら最大反復回数に達した（収束しきらず）
    # NOTE: もとのコードの print は 0 で割る式があり例外化するため、エラーメッセージ生成は簡素化推奨。
    print(f"警告: ガウス-ザイデル法が {max_iterations} 回以内に収束しませんでした。")
    u_next[:] = u_iter  # 現時点の反復解を採用
    return max_iterations


def animate_results(
    results: list, grid_space: float, time_delta: float, M: int
) -> None:
    """
    計算結果 (各タイムステップの温度分布) をアニメーションで可視化して GIF 保存。

    可視化のポイント:
      - y軸範囲を固定（-5〜105℃）してスケールの揺れを防止
      - プロファイルは点マーカー＋破線で変化が見やすい描画
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    x_values = np.arange(M + 1) * grid_space  # x座標配列
    ax.set_xlabel("Position X (m)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("1D Heat Conduction (Crank-Nicolson with Gauss-Seidel)")
    ax.grid(True)
    ax.set_ylim(-5, 105)  # y軸の範囲を固定

    frames = []  # 各フレームで描くアーティストの集合
    for j, profile in enumerate(results):
        time = j * time_delta
        (line,) = ax.plot(x_values, profile, linestyle="--", marker="o", color="b")
        time_text = ax.text(
            0.05,
            0.95,
            f"t = {time:.2f} h",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
        )
        frames.append([line, time_text])

    # 事前に生成したアーティスト群をアニメーション化
    # blit=True は効率が良いが、環境によってテキスト更新が不安定なら False に変更
    anim = ArtistAnimation(fig, frames, interval=100, blit=True, repeat_delay=1000)
    anim.save(ANIMATION_GIF_PATH, writer="pillow")
    plt.close(fig)  # メモリ解放
    print(f"アニメーションを {ANIMATION_GIF_PATH} に保存しました。")


def save_results_to_file(results: list, params: dict) -> None:
    """
    計算結果の数値データと使用したパラメータをテキストファイルに保存します。
    各時間ステップの温度分布を転置して（空間×時間の行列にして）保存。

    形式:
      - 先頭に Parameters（再現性のため）
      - その後に行列 U: 行=空間 i, 列=時間 j
    """
    # 結果（時刻ごとの1Dプロファイルのリスト）→ 2D行列に変換し、(空間, 時間) に転置
    u_result_matrix = np.array(results).T
    with open(RESULT_FILE_PATH, "w", encoding="utf-8") as file:
        file.write("# Calculated result of 1D Heat Conduction (Crank-Nicolson)\n\n")
        file.write("# Parameters:\n")
        for key, value in params.items():
            file.write(f"#   {key}: {value}\n")
        file.write("\n# Matrix U (u[space, time]):\n")
        # 小数点以下6桁で保存（解析ツールに読みやすい）
        np.savetxt(file, u_result_matrix, fmt="%.6f")
    print(f"計算結果を {RESULT_FILE_PATH} に保存しました。")


if __name__ == "__main__":
    """
    メイン実行ブロック。
    1次元熱伝導方程式をクランク-ニコルソンの半陰解法で時刻ごとに解き、
    各時刻の連立一次方程式をガウス-ザイデル反復で求解する。
    """
    # --- 計算パラメータの設定 ---
    params = {
        "grid_counts_x": 10,  # 空間方向の格子分割数 M（節点は 0..M）
        "grid_counts_t": 50,  # 時間方向のステップ数 N（時刻は 0..N）
        "grid_space": 0.1,  # Δx [m]
        "time_delta": 0.2,  # Δt [h]
        "max_iterations_gs": 200,  # ガウス-ザイデルの最大反復回数
        "thermal_diffusivity": 0.07,  # χ [m^2/h]
    }
    # --- 変数展開（可読性のため） ---
    M = params["grid_counts_x"]
    N = params["grid_counts_t"]
    dx = params["grid_space"]
    dt = params["time_delta"]
    max_iter_gs = params["max_iterations_gs"]
    chi = params["thermal_diffusivity"]

    # 無次元パラメータ R = χ Δt / Δx²
    R_chi = chi * dt / dx**2
    print(f"無次元パラメータ R*χ = {R_chi:.4f}")
    # 参考表示: 下は「χ=1 と見なしたときの見かけの R=Δt/Δx²」。
    # 明確な誤解を避けるなら χ を掛けた R_chi のほうを主に参照すること。
    print(f"(参考) 陽解法の安定性パラメータ R = {dt / dx**2:.4f}")

    # 時刻ごとのプロファイル（1D）を保持
    u_current = np.zeros(M + 1)  # 初期時刻 j=0（内部0, 境界はこの後設定）
    u_next = np.zeros(M + 1)

    # 2D表現の雛形関数を呼ぶ“デモ”的な行。実際の初期条件は上で u_current を 0 で満たしているため十分。
    set_initial_condition(np.zeros((M + 1, N + 1)))  # 参考呼び出し

    # 結果リストに初期状態を追加
    results = [u_current.copy()]

    # 時間発展ループ（j=0..N-1）
    for j in range(N):
        # 2D雛形の境界設定を“呼ぶだけ”のデモ。実際に使うのは下の1D境界代入。
        set_boundary_condition(np.zeros((M + 1, N + 1)))  # 参考呼び出し

        # 1Dの境界条件（Dirichlet）を次時刻解 u_next に適用
        u_next[0] = 0.0
        u_next[-1] = 100.0

        # Crank–Nicolson の 1 ステップを GS で解く
        iterations = calculate_crank_nicolson_step(
            u_next=u_next,
            u_current=u_current,
            M=M,
            R_chi=R_chi,
            max_iterations=max_iter_gs,
        )

        # 次時刻を現在に更新し、結果を保存
        u_current[:] = u_next
        results.append(u_current.copy())

        print(
            f"時間ステップ {j+1}/{N} 完了, "
            f"時刻 t = {(j+1)*dt:.2f} h, "
            f"ガウス-ザイデル反復回数: {iterations}"
        )

    # 可視化とファイル出力
    animate_results(results, grid_space=dx, time_delta=dt, M=M)
    save_results_to_file(results, params)
    print("すべての処理が完了しました。")
