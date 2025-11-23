# -*- coding: utf-8 -*-
# ============================================================
# 1次元熱伝導方程式（拡散方程式）の陽解法（FTCS）実装 + 可視化/保存
# ------------------------------------------------------------
# 【理論メモ（要点）】
# • 連続方程式（棒の温度 u(x,t)）:
#     ∂u/∂t = χ ∂²u/∂x²,  0≤x≤L, t≥0
#   ここで χ は温度伝導率（thermal diffusivity）。
#
# • 離散化（等間隔格子, Δx, Δt）:
#     時間: t_j = j Δt,   空間: x_i = i Δx
#     FTCS（Forward-Time, Central-Space）:
#       u_i^{j+1} = u_i^j + R (u_{i+1}^j - 2u_i^j + u_{i-1}^j)
#     ただし R = χ Δt / (Δx)^2  は無次元の安定性パラメータ。
#
# • 安定性（フォン・ノイマン解析）:
#     1次元FTCSは 0 ≤ R ≤ 1/2 で安定（R>1/2 で発散）。
#     本コードでは validate_parameters() で R≤0.5 を検証して防御。
#
# • 境界条件（Dirichlet）:
#     左端 x=0: u=0,  右端 x=L: u=max_temperature を全時刻で固定。
#     内部点だけを更新し、境界は毎時刻コピー（Dirichletの標準的取り扱い）。
#
# • 初期条件:
#     t=0 における内部温度を 0 とする（境界以外）。
#
# • 近似の精度:
#     時間方向 O(Δt), 空間方向 O((Δx)^2) の局所離散化誤差。
#     収束は R を満たし Δt, Δx → 0 で所望の連続解に近づく。
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import ArtistAnimation

# --- 定数定義 ---
# 生成するアニメーションGIFファイルのパス
ANIMATION_GIF_PATH = "./animation.gif"
# 計算結果を出力するテキストファイルのパス
RESULT_FILE_PATH = "./calculated_result.txt"


def set_initial_condition(u: np.ndarray) -> None:
    """
    初期条件 (時刻 t=0) を設定する。

    理論背景:
    - 初期値問題として、内部点の u(x,0) を与える必要がある。
    - ここでは「棒内部は0℃」という単純なステップ初期条件。
      （境界の温度は別関数で時刻全体にわたり固定する）

    Args:
        u (np.ndarray):
            温度分布を格納する2次元NumPy配列。
            形状は (空間格子点数 M+1, 時間ステップ数 N+1)。
            u[1:-1, 0] の要素が変更される（内部点の t=0）。
    """
    # u[空間点i, 時間点j]
    # 境界 (i=0 と i=M) を除く内部格子点 (i=1 から M-1) の
    # 時刻 j=0 における温度を 0.0 に設定。
    u[1:-1, 0] = 0.0


def set_boundary_condition(u: np.ndarray, max_temperature: float) -> None:
    """
    Dirichlet 境界条件（全時刻 t≥0）を設定する。

    理論背景:
    - FTCS では境界点は式の右辺に含めるだけであり、境界の値は毎時刻「既知値」として扱う。
      Dirichlet の場合、各時刻で端点を直接上書きするのが正道。

    Args:
        u (np.ndarray):
            温度分布 2D 配列。u[0, :] と u[-1, :] を時刻全体で設定する。
        max_temperature (float):
            右端 (x=L) に課す温度 [℃]（時間不変）。
    """
    # 左端（i=0）を全時刻で 0℃
    u[0, :] = 0.0
    # 右端（i=M, インデックス -1）を全時刻で max_temperature℃
    u[-1, :] = max_temperature


def calculate_diffusion_equation(
    M: int, N: int, R: float, max_temperature: float
) -> np.ndarray:
    """
    1次元熱伝導方程式を陽解法（FTCS）で時間前進させる。

    差分スキーム（内部点 i=1..M-1）:
        u[i, j+1] = (1 - 2R) * u[i, j] + R * (u[i+1, j] + u[i-1, j])

    安定性:
        R = χ Δt / (Δx)^2 で、0 ≤ R ≤ 1/2 が必要（1D）。

    Args:
        M (int): 空間方向の格子分割数（節点は M+1）
        N (int): 時間方向のステップ数（時間点は N+1）
        R (float): 安定性パラメータ（χ Δt / Δx^2）
        max_temperature (float): 右端の境界温度 [℃]

    Returns:
        np.ndarray: 温度分布 u（形状: (M+1, N+1)）
    """
    # 温度配列をゼロ初期化（列: 時間、行: 空間）
    u = np.zeros((M + 1, N + 1))

    # 初期条件と境界条件を設定
    set_initial_condition(u)
    set_boundary_condition(u, max_temperature)

    # 時間前進: j=0..N-1
    for j in range(N):
        # 内部点 i=1..M-1 をFTCSで更新
        # 注）境界は毎時刻すでに既知（Dirichlet）なので、ここでは触れない
        for i in range(1, M):
            # FTCS（Forward Time, Central Space）
            # u[i, j+1] = u[i, j] + R (u[i+1, j] - 2u[i, j] + u[i-1, j])
            #           = (1-2R) u[i,j] + R (u[i+1,j] + u[i-1,j])
            u[i, j + 1] = (1.0 - 2.0 * R) * u[i, j] + R * (u[i + 1, j] + u[i - 1, j])

        # もし時間依存の境界条件を扱う場合は、このループ末尾で
        # set_boundary_condition(u, ...) を呼ぶと「全時刻での境界の再設定」になる。
        # （本例は時間不変なので毎回の再設定は不要）

        # 【参考】ベクトル化（高速化）案（同じ計算式を一括評価）:
        # u[1:M, j+1] = (1-2*R)*u[1:M, j] + R*(u[2:M+1, j] + u[0:M-1, j])

    return u


def animate_results(u: np.ndarray, grid_space: float, time_delta: float) -> None:
    """
    計算結果 u(x_i, t_j) の時間発展をアニメーション（GIF）にする。

    可視化上の留意:
      • ArtistAnimation は「各フレームで更新する Artist のリスト」を受け取る設計。
      • 物理軸は x=iΔx（横軸）、縦軸は温度 u（縦軸）。
      • GIF の単位時間（frame interval）は可視化用の擬似時間。

    Args:
        u (np.ndarray): 温度分布（形状: (M+1, N+1)）
        grid_space (float): Δx [m]
        time_delta (float): Δt [h]（可視化テキスト表示用）
    """
    # 時間を先頭次元にする（フレームループを回しやすい）
    u_transposed = u.T  # 形状: (N+1, M+1)

    # x座標の物理位置（0, Δx, 2Δx, ...）
    x_values = np.arange(u.shape[0]) * grid_space  # 長さ: M+1

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlabel("Position X (m)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("1D Heat Conduction (Explicit FTCS)")
    ax.grid(True)

    frames = []
    for j, temperature_profile in enumerate(u_transposed):
        # 温度分布のプロット（折れ線 + マーカー）
        (line,) = ax.plot(
            x_values,
            temperature_profile,
            linestyle="--",
            marker="o",
            color="b",
        )
        # 時刻表示（可視化用の注記）
        time_text = ax.text(
            0.05,
            0.95,
            f"t = {j * time_delta:.2f} h",
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="top",
        )
        frames.append([line, time_text])

    # フレーム → アニメーション
    anim = ArtistAnimation(fig, frames, interval=100)
    anim.save(ANIMATION_GIF_PATH, writer="pillow")
    plt.close(fig)
    print(f"アニメーションを {ANIMATION_GIF_PATH} に保存しました。")


def save_results_to_file(u: np.ndarray, params: dict) -> None:
    """
    計算結果 u と使用パラメータをテキストに保存（再現性のために重要）。

    理論・実務の観点:
      • R, Δx, Δt, χ のセットアップは可換ではないため、記録がないと再現不能。
      • データは行=空間, 列=時間の行列として保存（視覚・解析の都合で選択可能）。

    Args:
        u (np.ndarray): 温度分布（形状: (M+1, N+1)）
        params (dict): 計算パラメータ
    """
    with open(RESULT_FILE_PATH, "w", encoding="utf-8") as file:
        file.write("# Calculated result of 1D Heat Conduction (Explicit Method)\n\n")
        file.write("# Parameters:\n")
        for key, value in params.items():
            file.write(f"#   {key}: {value}\n")
        file.write("\n# Matrix U (u[space, time]):\n")
        # 数値行列の保存（%.6f で等幅っぽくし読みやすさ確保）
        np.savetxt(file, u, fmt="%.6f")
    print(f"計算結果を {RESULT_FILE_PATH} に保存しました。")


def validate_parameters(R: float) -> None:
    """
    安定性条件（R ≤ 0.5）を検証。違反時は例外を送出。

    理論背景（フォン・ノイマン法）:
      • 1D FTCS の増幅因子 G(k) = 1 - 4R sin²(kΔx/2)
      • 安定のためには |G(k)| ≤ 1 が全 k で必要 ⇒ 0 ≤ R ≤ 1/2

    Args:
        R (float): R = χ Δt / (Δx)^2

    Raises:
        ValueError: R>0.5 の場合（Δt を小さく/Δx を大きく等の調整を促す）
    """
    if R > 0.5:
        error_message = (
            f"安定性条件を満たしていません (R = {R:.4f} > 0.5)。\n"
            f"time_delta (Δt) を小さくするか、grid_space (Δx) を大きくしてください。"
        )
        raise ValueError(error_message)


if __name__ == "__main__":
    """
    メイン実行ブロック。
    • 手順:
        1) パラメータ設定  →  2) R 計算 & 安定性チェック
        3) 計算（FTCS）    →  4) アニメーション出力
        5) 数値とパラメータの保存
    • 参考:
        L = M * Δx を 1 とみなす場合は、(M, Δx) の組を調整して L=1 を満たせばよい。
    """
    # --- 計算パラメータの設定 ---
    params = {
        "grid_counts_x": 10,  # M: 空間分割数（節点は M+1）
        "grid_counts_t": 150,  # N: 時間ステップ数（時間点は N+1）
        "grid_space": 0.1,  # Δx [m]
        "time_delta": 0.07,  # Δt [h]
        "thermal_diffusivity": 0.07,  # χ [m^2/h]
        "max_temperature": 100.0,  # 右端境界温度 [℃]
    }
    # --- R の計算（χ Δt / Δx^2）
    R = (
        params["thermal_diffusivity"]
        * params["time_delta"]
        / (params["grid_space"] ** 2)
    )
    print(f"安定性パラメータ R = {R:.4f}")

    try:
        # 安定性チェック
        validate_parameters(R)

        print("計算を開始します...")
        # 拡散方程式（FTCS）を実行
        u_result = calculate_diffusion_equation(
            M=params["grid_counts_x"],
            N=params["grid_counts_t"],
            R=R,
            max_temperature=params["max_temperature"],
        )
        print("計算が完了しました。")

        print("結果をアニメーション化・保存します...")
        animate_results(
            u_result,
            grid_space=params["grid_space"],
            time_delta=params["time_delta"],
        )

        # 結果とパラメータを保存（再現性確保）
        save_results_to_file(u_result, params)

        print("すべての処理が完了しました。")
    except ValueError as e:
        # R の検証エラー（安定性を外れた設定）
        print(f"エラー: {e}")
