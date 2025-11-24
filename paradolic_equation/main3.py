import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import ArtistAnimation

# 日本語フォント設定（ユーザー要望に合わせた可読性向上用。環境にフォントが無い場合は無視されます）
from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = "Meiryo"


def _set_initial_condition(
    u_array: list[list[list[float]]],
    m: int,
    n: int,
) -> None:
    """
    初期条件 (t = 0) を設定する関数。
    板（2Dグリッド）の内部格子点 (1 ≤ i < m, 1 ≤ j < n) に対し、初期温度を 0.0 に設定する。

    Parameters
    ----------
    u_array : list[list[list[float]]]
        温度データを格納する 3 次元リスト。添字は [time k][x i][y j]。
        サイズ：時間 (l+1) × (m+1) × (n+1) を想定。
    m : int
        x 方向の格子分割数（格子点は 0..m の計 m+1 個）。
    n : int
        y 方向の格子分割数（格子点は 0..n の計 n+1 個）。

    Notes
    -----
    - 境界 (i=0, i=m, j=0, j=n) は境界条件で与えるため、内部点のみを設定する。
    - 初期時刻は k=0 固定（u_array[0] が t=0 に対応）。
    """
    # 境界を除く内部格子点 (1..m-1, 1..n-1) の初期温度を 0.0 に設定
    for i in range(1, m):
        for j in range(1, n):
            u_array[0][i][j] = 0.0


def _set_boundary_condition(
    u_array: list[list[list[float]]],
    m: int,
    n: int,
    l: int,
    dx: float,
) -> None:
    """
    Dirichlet 型の境界条件を、全時間ステップに対して設定する関数。

    本例の境界条件：
      - y = 0（下端）: u = 0
      - y = 1（上端）: u = 4 x (1 - x)  （x∈[0,1]で最大1、放物線）
      - x = 0（左端）: u = 0
      - x = 1（右端）: u = 0

    Parameters
    ----------
    u_array : list[list[list[float]]]
        温度データ 3 次元リスト [k][i][j]。
    m : int
        x 方向の格子分割数。
    n : int
        y 方向の格子分割数。
    l : int
        時間ステップ数（k は 0..l）。
    dx : float
        x 方向の格子間隔（上端境界の x を実数に変換するために使用）。
    """
    # すべての時間ステップに対して境界値を設定
    for k in range(l + 1):
        # y=0（下端）と y=n（上端）
        for i in range(m + 1):
            u_array[k][i][0] = 0.0  # 下端 (y=0) で u=0
            x = i * dx
            u_array[k][i][n] = 4.0 * x * (1.0 - x)  # 上端 (y=1) で u=4x(1-x)

        # x=0（左端）と x=m（右端）
        for j in range(n + 1):
            u_array[k][0][j] = 0.0  # 左端 (x=0) で u=0
            u_array[k][m][j] = 0.0  # 右端 (x=1) で u=0


def calculate_temperature_distribution(
    m: int,
    n: int,
    l: int,
    dx: float,
    dy: float,
    dt: float,
    chi: float,
) -> list[list[list[float]]]:
    """
    2 次元熱伝導方程式（拡散方程式）を **陽解法（FTCS, Forward-Time Centered-Space）** で解く。

    離散化（内部格子点 i=1..m-1, j=1..n-1）：
        u[k+1][i][j] = u[k][i][j] + χ Δt [ (u[k][i+1][j] - 2 u[k][i][j] + u[k][i-1][j]) / Δx²
                                            + (u[k][i][j+1] - 2 u[k][i][j] + u[k][i][j-1]) / Δy² ]

    Parameters
    ----------
    m, n : int
        それぞれ x, y 方向の格子分割数（点は 0..m, 0..n）。
    l : int
        時間ステップ数（k は 0..l）。
    dx, dy : float
        x, y 方向の格子間隔。
    dt : float
        時間刻み幅。
    chi : float
        温度伝導率（拡散係数） χ [m^2/h]。

    Returns
    -------
    list[list[list[float]]]
        温度場 u[k][i][j]（サイズ： (l+1) × (m+1) × (n+1) ）を格納した 3 次元リスト。

    Notes
    -----
    - **安定性**（CFL 条件の一種）：
        χ Δt (1/Δx² + 1/Δy²) ≤ 1/2 を満たす必要あり（validate_stability で確認可能）。
      条件を破ると、陽解法は発散しうる。
    - **計算量**：O(l · m · n)。リスト実装のため、NumPy ベクトル化に比べると遅いが、可読性を優先。
    """
    # 3D リストを初期化：時間 (l+1) × x (m+1) × y (n+1)
    u = [[[0.0 for _ in range(n + 1)] for _ in range(m + 1)] for _ in range(l + 1)]

    # 初期条件と境界条件を設定
    _set_initial_condition(u_array=u, m=m, n=n)
    _set_boundary_condition(u_array=u, m=m, n=n, l=l, dx=dx)

    # --- 時間ステップループ（k = 0 .. l-1） ---
    for k in range(l):
        # 内部格子点 (i=1..m-1, j=1..n-1) を FTCS で更新
        for i in range(1, m):
            for j in range(1, n):
                # x 方向の 2 階中央差分（離散ラプラシアンの x 成分）
                #   (u[k][i+1][j] - 2*u[k][i][j] + u[k][i-1][j]) / dx^2
                term_x = (u[k][i + 1][j] - 2.0 * u[k][i][j] + u[k][i - 1][j]) / (dx**2)

                # y 方向の 2 階中央差分（離散ラプラシアンの y 成分）
                #   (u[k][i][j+1] - 2*u[k][i][j] + u[k][i][j-1]) / dy^2
                term_y = (u[k][i][j + 1] - 2.0 * u[k][i][j] + u[k][i][j - 1]) / (dy**2)

                # 陽解法（Forward Euler in time）で次時刻へ前進
                #   u^{k+1} = u^k + χΔt ( term_x + term_y )
                u[k + 1][i][j] = u[k][i][j] + chi * dt * (term_x + term_y)

        # 注: 境界は全時刻で固定済み（Dirichlet）。内部更新後も境界はそのまま。

    return u


def animate_time_evolution(
    u_array: list[list[list[float]]],
    m: int,
    n: int,
    l: int,
    dx: float,
    dy: float,
    dt: float,
    output_filename: str = "animation.gif",
) -> None:
    """
    計算した 2D 温度場 u[k][i][j] の時間発展を GIF で保存する。

    Parameters
    ----------
    u_array : list[list[list[float]]]
        温度場データ（時間×x×y）。
    m, n : int
        x, y 方向の分割数。
    l : int
        時間ステップ数。
    dx, dy : float
        格子間隔（表示範囲の物理スケールに反映）。
    dt : float
        時間刻み幅（キャプションに使用）。
    output_filename : str
        出力 GIF ファイル名。

    Notes
    -----
    - `imshow` は (row, col)=(y, x) で解釈するため、u[k][i][j]（i: x, j: y）を転置して描画。
    - カラーバーの上限 v_max は上端境界 u=4x(1-x) の最大値 1.0 に合わせている。
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # 画像の物理範囲を指定（格子点がセルの中心に来るように半セルずらす）
    extent = [
        0.0 - dx * 0.5,
        m * dx + dx * 0.5,
        0.0 - dy * 0.5,
        n * dy + dy * 0.5,
    ]
    ax.set_xlabel("Position X (m)")
    ax.set_ylabel("Position Y (m)")

    frames = []
    for k in range(l + 1):
        # u_array[k] は (m+1)×(n+1) の [i][j] 配列。imshow 用に (j,i) へ転置。
        u_2d_time_k = [[u_array[k][i][j] for j in range(n + 1)] for i in range(m + 1)]

        frame = ax.imshow(
            np.array(u_2d_time_k).T,  # 転置して (y, x) にする
            cmap="viridis",  # カラーマップ（任意）
            interpolation="nearest",  # 補間なし（格子が見やすい）
            aspect="auto",  # 図のサイズに応じ自動調整
            origin="lower",  # y=0 を下に
            extent=extent,  # 物理座標範囲
            vmin=0.0,  # 下端境界 0 に合わせる
            vmax=1.0,  # 上端境界の最大値 1.0 に合わせる
        )

        # 時刻ラベルを軸座標系で表示
        current_time = k * dt
        time_text = ax.text(
            0.05,
            1.02,
            f"Time = {current_time:.3f} h",
            transform=ax.transAxes,
            fontsize=12,
        )

        # カラーバーは最初のフレームだけ追加（重複防止）
        if k == 0:
            cbar = fig.with_scraped if False else fig.colorbar(frame, ax=ax)
            cbar.set_label("Temperature (a.u.)")  # a.u. = arbitrary unit

        frames.append([frame, time_text])

    # blit=True は高速だが、環境によってはテキストが正しく更新されない場合がある
    anim = ArtistAnimation(
        fig,
        frames,
        interval=50,  # 1 フレーム当たり 50ms
        blit=True,
        repeat_delay=1000,  # ループ再開前に 1 秒待機
    )

    # GIF で保存（要: pillow）
    anim.save(output_filename, writer="pillow")
    plt.close(fig)  # メモリ解放
    print(f"アニメーションを保存しました: {output_filename}")


def output_result_file(
    u_array: list[list[list[float]]],
    m: int,
    n: int,
    l: int,
    dx: float,
    dy: float,
    dt: float,
    chi: float,
    output_filename: str = "calculated_result.txt",
) -> None:
    """
    計算パラメータと各時刻の温度行列をテキストファイルに保存する。

    Parameters
    ----------
    u_array : list[list[list[float]]]
        温度場データ（時間×x×y）。
    m, n, l : int
        空間分割・時間ステップ数。
    dx, dy : float
        格子間隔。
    dt : float
        時間刻み幅。
    chi : float
        温度伝導率 χ。
    output_filename : str
        出力ファイル名。

    出力形式（例）
    -------------
    # 2D Heat Conduction (Explicit Method) - Calculation Result

    # Calculation Parameters:
    #   Grid Size (x, y): 10 x 10
    #   Time Steps (t): 500
    #   Grid Spacing (dx, dy): 0.100 m, 0.100 m
    #   Time Step (dt): 0.0020 h
    #   Thermal Diffusivity (chi): 0.0700 m^2/h

    # Calculated Temperature Matrix U[t][x][y] at each time step:

    # Time = 0.0000 h
    <(m+1)×(n+1) の行列を np.savetxt で出力>
    ...
    """
    with open(output_filename, "w", encoding="utf-8") as file:
        file.write("# 2D Heat Conduction (Explicit Method) - Calculation Result\n\n")
        file.write("# Calculation Parameters:\n")
        file.write(f"#   Grid Size (x, y): {m} x {n}\n")
        file.write(f"#   Time Steps (t): {l}\n")
        file.write(f"#   Grid Spacing (dx, dy): {dx:.3f} m, {dy:.3f} m\n")
        file.write(f"#   Time Step (dt): {dt:.4f} h\n")
        file.write(f"#   Thermal Diffusivity (chi): {chi:.4f} m^2/h\n\n")
        file.write("# Calculated Temperature Matrix U[t][x][y] at each time step:\n")

        for k in range(l + 1):
            current_time = k * dt
            file.write(f"\n# Time = {current_time:.4f} h\n")
            # u_array[k] は (m+1)×(n+1) の [i][j]。そのまま書き出して可視性を優先。
            u_2d_time_k = [
                [u_array[k][i][j] for j in range(n + 1)] for i in range(m + 1)
            ]
            np.savetxt(file, np.array(u_2d_time_k), fmt="%.6f")

    print(f"計算結果をファイルに保存しました: {output_filename}")


def validate_stability(chi: float, dt: float, dx: float, dy: float) -> None:
    """
    陽解法（FTCS）の安定性条件（CFL 条件）を検証するユーティリティ。

    条件：
        χ Δt (1/Δx² + 1/Δy²) ≤ 0.5

    Parameters
    ----------
    chi : float
        温度伝導率 χ。
    dt : float
        時間刻み幅 Δt。
    dx, dy : float
        格子間隔 Δx, Δy。

    Raises
    ------
    ValueError
        条件を満たさない場合（発散の恐れが高い）。
    """
    stability_value = chi * dt * (1.0 / dx**2 + 1.0 / dy**2)

    if stability_value > 0.5:
        error_message = (
            "安定性条件を満たしていません。\n"
            f"  χΔt(1/Δx² + 1/Δy²) = {stability_value:.4f} > 0.5\n"
            f"時間刻み幅 dt ({dt:.4f}) を小さくするか、"
            f"格子間隔 dx ({dx:.3f}), dy ({dy:.3f}) を大きくしてください。"
        )
        raise ValueError(error_message)
    else:
        print(f"安定性条件 OK (値: {stability_value:.4f} <= 0.5)")


if __name__ == "__main__":
    """メイン実行ブロック：2D 熱伝導（陽解法）シミュレーションの一括実行"""

    # --- パラメータ設定 ---
    m: int = 10  # x 方向の格子分割数（点は 0..m）
    n: int = 10  # y 方向の格子分割数（点は 0..n）
    l: int = 500  # 時間ステップ数（k は 0..l）

    dx: float = 0.1  # Δx [m]
    dy: float = 0.1  # Δy [m]

    # 時間刻み幅 Δt [h]
    # 安定性目安：0.07 * dt * (1/0.1^2 + 1/0.1^2) <= 0.5
    #            → dt <= 0.5 / (0.07 * 200) ≈ 0.0357...
    dt: float = 0.002

    chi: float = 0.07  # 温度伝導率（拡散係数）χ [m^2/h]

    try:
        # 1) 安定性条件をまずチェック（重要）
        validate_stability(chi=chi, dt=dt, dx=dx, dy=dy)

        # 2) 温度分布の時間発展を計算
        print("計算を開始します...")
        u_result = calculate_temperature_distribution(
            m=m, n=n, l=l, dx=dx, dy=dy, dt=dt, chi=chi
        )
        print("計算が完了しました。")

        # 3) アニメーションの作成・保存（上端の放物線境界により彩色が 0..1 の範囲に収まる想定）
        print("アニメーションを作成中...")
        animate_time_evolution(u_array=u_result, m=m, n=n, l=l, dx=dx, dy=dy, dt=dt)

        # 4) 数値結果（各時刻の行列）とパラメータをファイル出力
        print("結果をファイルに出力中...")
        output_result_file(
            u_array=u_result, m=m, n=n, l=l, dx=dx, dy=dy, dt=dt, chi=chi
        )

        print("処理が正常に終了しました。")

    except ValueError as ve:
        # 安定性条件違反などのエラー
        print(f"\nエラーが発生しました:\n{ve}")
    except Exception as e:
        # その他の予期せぬエラー
        print(f"\n予期せぬエラーが発生しました: {e}")
