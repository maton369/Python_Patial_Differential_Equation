import os
from dataclasses import dataclass
from typing import Optional  # ← Python 3.10 互換: Self は使わない

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import ArtistAnimation
from pydantic import BaseModel, Field

# （表示の日本語フォント設定：ユーザ希望に合わせて挿入）
from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = "Meiryo"

# --- 定数定義 ---
DEFAULT_ERROR_TOLERANCE: float = 1.0e-4  # 収束判定の許容誤差（GS 反復の停止基準）
DEFAULT_INITIAL_VALUE: float = 0.0  # 内部格子点の初期値（t=0 の内部温度）
DEFAULT_OUTPUT_DIR: str = "./output"  # 出力ディレクトリ
DEFAULT_ANIMATION_FILENAME: str = "animation.gif"  # アニメーションのファイル名
DEFAULT_RESULT_FILENAME: str = "calculated_result.txt"  # 結果ファイル名


class GridPoint(BaseModel):
    """格子点の値を表す Pydantic モデル。
    Attributes:
        value (float): 格子点の値（温度など）。デフォルトは 0.0。
    """

    value: float = Field(
        default=DEFAULT_INITIAL_VALUE, description="格子点の値（温度など）"
    )


class Solution(BaseModel):
    """方程式の解（温度分布）を表す Pydantic モデル。
    2次元格子（空間 × 時間）の各点の値を GridPoint の 2 次元リストとして保持します。

    Attributes:
        grid (list[list[GridPoint]]): 2次元格子上の値を保持するリスト。
            形状は (M+1 行, N+1 列) を想定（行=空間 i, 列=時間 j）。
    """

    grid: list[list[GridPoint]] = Field(
        default_factory=list, description="2次元格子上の値"
    )

    @classmethod
    def create_empty(
        cls, x_size: int, y_size: int
    ) -> "Solution":  # ← Self を文字列参照に変更
        """指定されたサイズの空のソリューション（全格子点デフォルト値）を作成します。
        Args:
            x_size (int): x方向の格子点数 (M+1)。
            y_size (int): y方向（時間）の格子点数 (N+1)。
        Returns:
            Solution: 初期化された Solution オブジェクト。
        """
        grid = [[GridPoint() for _ in range(y_size)] for _ in range(x_size)]
        return cls(grid=grid)

    def get_value(self, i: int, j: int) -> float:
        """指定された格子点 (i, j) の値を取得します。
        Args:
            i (int): x方向のインデックス（0..M）。
            j (int): y方向（時間）のインデックス（0..N）。
        Returns:
            float: 格子点の値。
        """
        return self.grid[i][j].value

    def set_value(self, i: int, j: int, value: float) -> None:
        """指定された格子点 (i, j) に値を設定します。
        Args:
            i (int): x方向のインデックス。
            j (int): y方向（時間）のインデックス。
            value (float): 設定する値。
        """
        self.grid[i][j].value = value

    def to_list(self) -> list[list[float]]:
        """Solution を値のみの 2 次元リストに変換します。
        主に NumPy 配列への変換やファイル出力に使用します。
        Returns:
            list[list[float]]: 値のみの 2 次元リスト。
        """
        return [[point.value for point in row] for row in self.grid]

    @classmethod
    def from_list(
        cls, data: list[list[float]]
    ) -> "Solution":  # ← Self を文字列参照に変更
        """値のみの 2 次元リストから Solution を作成します。
        主に Solution のディープコピー代替として使用します。
        Args:
            data (list[list[float]]): 値の 2 次元リスト。
        Returns:
            Solution: 生成された Solution オブジェクト。
        """
        grid = [[GridPoint(value=val) for val in row] for row in data]
        return cls(grid=grid)


@dataclass
class SimulationConfig:
    """シミュレーション設定を管理するデータクラス。
    dataclass を使用して、設定項目とその型を明示的に定義します。
    __post_init__ で基本的なバリデーションも行います。

    Attributes:
        grid_counts_x (int): 空間方向の格子分割数 M（点は M+1）。
        grid_counts_t (int): 時間方向のステップ数 N（点は N+1）。
        grid_space (float): 空間方向の格子間隔 Δx [m]。
        time_delta (float): 時間方向の刻み幅 Δt [h]。
        max_iterations (int): ガウス-ザイデル法の最大反復回数。
        thermal_diffusivity (float): 温度伝導率 χ [m^2/h]。
        error_tolerance (float): 収束判定の相対許容誤差。
        initial_value (float): 内部格子点の初期値。
        output_dir (str): 結果ファイルの出力ディレクトリ。
    """

    grid_counts_x: int
    grid_counts_t: int
    grid_space: float
    time_delta: float
    max_iterations: int
    thermal_diffusivity: float
    error_tolerance: float = DEFAULT_ERROR_TOLERANCE
    initial_value: float = DEFAULT_INITIAL_VALUE
    output_dir: str = DEFAULT_OUTPUT_DIR

    def __post_init__(self) -> None:
        """インスタンス生成後に行われる設定値の基本的な検証。"""
        if self.grid_counts_x <= 0 or self.grid_counts_t <= 0:
            raise ValueError(
                "格子数 (grid_counts_x, grid_counts_t) は正の整数である必要があります"
            )
        if self.grid_space <= 0 or self.time_delta <= 0:
            raise ValueError(
                "格子間隔 (grid_space) と時間刻み幅 (time_delta) は正の値である必要があります"
            )
        if self.max_iterations <= 0:
            raise ValueError(
                "最大反復回数 (max_iterations) は正の整数である必要があります"
            )
        if self.error_tolerance <= 0:
            raise ValueError("許容誤差 (error_tolerance) は正の値である必要があります")
        if self.thermal_diffusivity <= 0:
            raise ValueError(
                "温度伝導率 (thermal_diffusivity) は正の値である必要があります"
            )
        # 出力ディレクトリが存在しない場合は作成（冪等）
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def r_value(self) -> float:
        """r = Δt / (Δx)^2（注意: R_chi と別）を返します。
        Returns:
            float: r 値。
        """
        return self.time_delta / (self.grid_space**2)

    @property
    def r_chi_value(self) -> float:
        """R_chi = χ * Δt / (Δx)^2（理論での R に相当）を返します。
        Returns:
            float: R_chi 値。
        """
        return self.thermal_diffusivity * self.time_delta / (self.grid_space**2)


class ParabolicEquationSolver:
    """放物形方程式（1次元熱伝導）を Crank–Nicolson（半陰）+ Gauss–Seidel で解くクラス。
    - 初期条件/境界条件の適用
    - 各時刻ステップでの GS 反復
    - 収束判定（相対 L1 基準：改良余地あり）
    """

    def __init__(self, config: SimulationConfig):
        """ソルバーの初期化。
        Args:
            config (SimulationConfig): シミュレーション設定。
        """
        self.config = config
        self.solution: Optional[Solution] = None  # 計算実行後に格納

    def _initialize_solution(self) -> None:
        """計算に使用する Solution（空の 2D グリッド）を初期化します。"""
        # 格子点数は分割数+1（端点を含む）
        x_points = self.config.grid_counts_x + 1
        t_points = self.config.grid_counts_t + 1
        self.solution = Solution.create_empty(x_points, t_points)

    def _set_initial_condition(self) -> None:
        """t=0 の内部点に初期値（DEFAULT_INITIAL_VALUE）を設定します。"""
        if self.solution is None:
            raise RuntimeError("Solutionオブジェクトが初期化されていません。")
        # 端点（i=0, i=M）は境界条件で上書きするため、内部のみ設定
        for i in range(1, self.config.grid_counts_x):
            self.solution.set_value(i, 0, self.config.initial_value)

    def _set_boundary_condition(self) -> None:
        """全時刻の端点に Dirichlet 境界（左=0, 右=100）を設定します。"""
        if self.solution is None:
            raise RuntimeError("Solutionオブジェクトが初期化されていません。")
        # 左端 x=0 は 0℃
        for j in range(self.config.grid_counts_t + 1):
            self.solution.set_value(0, j, 0.0)
        # 右端 x=M は 100℃
        for j in range(self.config.grid_counts_t + 1):
            self.solution.set_value(self.config.grid_counts_x, j, 100.0)

    def _is_converged(
        self, current_solution: Solution, previous_solution: Solution, j: int
    ) -> bool:
        """GS 反復の収束判定（相対誤差：内部点の L1 比）。
        Args:
            current_solution (Solution): 現在反復の解（k+1）。
            previous_solution (Solution): 前反復の解（k）。
            j (int): 現在の時間ステップ。
        Returns:
            bool: True なら収束。
        """
        error_sum: float = 0.0
        value_sum: float = 0.0
        # 内部格子点 (i=1..M-1) の時刻 j+1 を比較
        for i in range(1, self.config.grid_counts_x):
            current_val = current_solution.get_value(i, j + 1)
            previous_val = previous_solution.get_value(i, j + 1)
            value_sum += abs(current_val)
            error_sum += abs(current_val - previous_val)

        # すべてがほぼ 0 の場合は「十分収束」とみなす（ゼロ除算防止）
        if value_sum < 1.0e-10:
            return True

        relative_error = error_sum / value_sum
        return relative_error <= self.config.error_tolerance

    def solve(self) -> Solution:
        """放物形方程式の数値解を計算します（CN+GS）。
        Returns:
            Solution: 計算された解（空間×時間の温度分布）。
        """
        # --- 初期化と条件設定 ---
        self._initialize_solution()
        if self.solution is None:  # 型チェッカ用ガード
            raise RuntimeError("Solutionの初期化に失敗しました。")
        self._set_initial_condition()
        self._set_boundary_condition()

        # 無次元パラメータ（理論の R=χΔt/Δx^2）
        r_chi = self.config.r_chi_value
        m = self.config.grid_counts_x
        q = 0.0  # 本例では生成項 Q=0（右辺にソースなし）

        # --- 時間ステップループ ---
        for j in range(self.config.grid_counts_t):
            # 次時刻 j+1 の初期推定値として「直前の時刻 j の値」を使用（GS 初期化）
            for i in range(1, m):
                self.solution.set_value(i, j + 1, self.solution.get_value(i, j))

            # --- ガウス-ザイデル反復ループ ---
            for iteration in range(self.config.max_iterations):
                # 反復前のスナップショット（k 反復の場）を確保
                previous_solution_iter = Solution.from_list(self.solution.to_list())

                # 内部点 i=1..m-1 を順に更新（左隣は新値，右隣は旧値 → GS）
                for i in range(1, m):
                    # 係数化（理論式を -2/R で正規化した形に対応）
                    # BS は A の対角項に対応する定数（時刻ステップ内で一定）
                    bs = -2.0 * (1.0 + 1.0 / r_chi)
                    # BL は右辺 b^j の i 成分（旧時刻 j のラプラシアン寄与などを含む）
                    bl = -(
                        previous_solution_iter.get_value(i + 1, j)
                        - 2.0 * previous_solution_iter.get_value(i, j)
                        + previous_solution_iter.get_value(i - 1, j)
                    ) - (2.0 / r_chi) * (previous_solution_iter.get_value(i, j) + q)

                    # 更新式（右隣は旧反復の値、左隣は今回更新済みの値）
                    # U_{i,j+1}^{(k+1)} = (BL - U_{i+1,j+1}^{(k)} - U_{i-1,j+1}^{(k+1)}) / BS
                    new_value = (
                        bl
                        - previous_solution_iter.get_value(
                            i + 1, j + 1
                        )  # 右隣：旧反復(k)
                        - self.solution.get_value(i - 1, j + 1)  # 左隣：新反復(k+1)
                    ) / bs
                    self.solution.set_value(i, j + 1, new_value)

                # --- 収束判定（相対 L1）---
                if self._is_converged(self.solution, previous_solution_iter, j):
                    print(f"時間ステップ {j+1}: {iteration+1} 回で収束しました")
                    break
            else:
                # break されなかった → 収束せず最大反復に達した
                print(
                    f"警告: 時間ステップ {j+1} で最大反復回数（{self.config.max_iterations}）に達しました。"
                    f" 収束していない可能性があります。"
                )

        # --- 全時間ステップ終了 ---
        return self.solution


class ResultVisualizer:
    """計算結果の可視化とファイル保存を担当するクラス（関心の分離）。"""

    def __init__(self, config: SimulationConfig):
        """ビジュアライザの初期化。
        Args:
            config (SimulationConfig): シミュレーション設定。
        """
        self.config = config

    def create_animation(self, solution: Solution) -> None:
        """温度分布の時間発展をアニメーション（GIF）に保存します。
        Args:
            solution (Solution): 計算された解オブジェクト（空間×時間）。
        """
        filename = os.path.join(self.config.output_dir, DEFAULT_ANIMATION_FILENAME)

        # Solution → ndarray（行=空間 i, 列=時間 j）に変換し、転置して時系列で走査しやすくする
        solution_array = np.array(solution.to_list()).T  # 形状: (N+1, M+1)
        x_values = np.arange(self.config.grid_counts_x + 1) * self.config.grid_space

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_xlabel("Position X (m)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("1D Heat Conduction (Crank-Nicolson OOD)")
        ax.grid(True)
        ax.set_ylim(-5, 105)  # y軸範囲を固定して比較しやすくする

        frames = []
        for j, profile in enumerate(solution_array):
            time = j * self.config.time_delta
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

        # blit=True は高速だが、環境によりテキスト更新で乱れる場合は False を検討
        anim = ArtistAnimation(fig, frames, interval=100, blit=True, repeat_delay=1000)
        anim.save(filename, writer="pillow")
        plt.close(fig)  # メモリ解放
        print(f"アニメーションを保存しました: {filename}")

    def save_result_to_file(self, solution: Solution) -> None:
        """数値結果とパラメータをテキストに保存します。
        Args:
            solution (Solution): 計算された解オブジェクト。
        """
        filename = os.path.join(self.config.output_dir, DEFAULT_RESULT_FILENAME)
        solution_array = np.array(solution.to_list())  # (M+1, N+1)

        with open(filename, "w", encoding="utf-8") as file:
            file.write("# Calculated result of 1D Heat Conduction (C-N OOD)\n\n")
            file.write("# Parameters:\n")
            # 必要に応じて dataclasses.asdict(config) で一括出力も可
            file.write(f"#   grid_counts_x: {self.config.grid_counts_x}\n")
            file.write(f"#   grid_counts_t: {self.config.grid_counts_t}\n")
            file.write(f"#   grid_space: {self.config.grid_space}\n")
            file.write(f"#   time_delta: {self.config.time_delta}\n")
            file.write(f"#   max_iterations: {self.config.max_iterations}\n")
            file.write(f"#   thermal_diffusivity: {self.config.thermal_diffusivity}\n")
            file.write(f"#   error_tolerance: {self.config.error_tolerance}\n")
            file.write(f"#   initial_value: {self.config.initial_value}\n")
            file.write("\n# Matrix U (u[space, time]):\n")
            np.savetxt(file, solution_array, fmt="%.6f")
        print(f"計算結果をファイルに保存しました: {filename}")


def run_simulation(config: SimulationConfig) -> None:
    """シミュレーションの実行フローを制御します。
    - ソルバーで解を計算
    - 可視化と保存を実施
    Args:
        config (SimulationConfig): シミュレーション設定。
    """
    print("--- シミュレーション開始 ---")

    # 1. 解く
    print("方程式を解いています...")
    solver = ParabolicEquationSolver(config)
    solution = solver.solve()
    print("計算完了。")

    # 2. 可視化と保存
    print("結果を処理中...")
    visualizer = ResultVisualizer(config)
    visualizer.create_animation(solution)
    visualizer.save_result_to_file(solution)

    print("--- シミュレーション完了 ---")


def main() -> None:
    """メイン関数: プログラムのエントリーポイント。"""
    try:
        # シミュレーション設定オブジェクトを作成
        config = SimulationConfig(
            grid_counts_x=10,
            grid_counts_t=50,
            grid_space=0.1,
            time_delta=0.2,
            max_iterations=200,
            thermal_diffusivity=0.07,
            output_dir="./cn_ood_output",  # 出力先ディレクトリ指定
        )
        # シミュレーション実行
        run_simulation(config)
    except ValueError as ve:
        print(f"設定エラー: {ve}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


if __name__ == "__main__":
    main()
