# -*- coding: utf-8 -*-
# ============================================================
# 楕円型PDE（2次元ラプラス方程式）を有限差分法 + ガウス–ザイデル法で解くOOP実装
# ------------------------------------------------------------
# 理論的背景（要点）:
#  1) 問題設定（連続系）
#     ∇^2 u = u_xx + u_yy = 0  （Ω 内, Dirichlet 境界条件）
#     本コードでは正方領域 [0, Lx]×[0, Ly]（Lx = grid_space * grid_counts_x 等）を想定し、
#     下・左・右の3辺で u=0、上辺 y=Ly で u(x,Ly)=f(x)=4x(1-x) を課す。
#
#  2) 空間離散化（有限差分, 等間隔格子, 格子幅 h=grid_space）
#     節点 (x_i, y_j) = (i h, j h), i=0..M, j=0..N（M=grid_counts_x, N=grid_counts_y）
#     5点ラプラシアン:
#         (u_{i+1,j} - 2u_{i,j} + u_{i-1,j})/h^2 + (u_{i,j+1} - 2u_{i,j} + u_{i,j-1})/h^2 = 0
#      ⇔  内点の更新式（ラプラス方程式で右辺0）:
#         u_{i,j} = (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1}) / 4
#     一貫性は局所誤差 O(h^2)（テイラー展開より）。滑らかな解に対する大域誤差も概ね O(h^2) を期待。
#
#  3) 反復解法（ガウス–ザイデル, Gauss–Seidel）
#     係数行列 A は M行列（離散最大値原理が成り立つ）に対応し、GS は収束。
#     低周波誤差の減衰が遅いため格子細分で反復回数が増えやすい（SOR/MG で高速化可）。
#
#  4) 停止判定
#     本実装は「反復差（U^{k+1}-U^k）」の相対量で停止。
#     これは PDE の満足度（残差 ||AU-b||）を直接保証しないため、厳密性が必要なら
#     離散残差ノルム（例: L∞）の併用が望ましい（実務上の推奨）。
# ============================================================

import os
from dataclasses import dataclass
from typing import Optional

# --- Self 型ヒント互換対策（Python 3.10 では typing に Self が無い）---
try:
    from typing import Self  # Python 3.11+
except ImportError:
    from typing_extensions import Self  # Python 3.8–3.10 用

import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel  # Pydanticモデルの基底クラス
from pydantic import Field  # フィールド定義/検証

# --- 定数定義 ---
# 定数を一箇所にまとめることで保守性↑。理論面では、error_tolerance は反復差に対する閾値（無次元）。
DEFAULT_ERROR_TOLERANCE: float = (
    1.0e-4  # 収束判定の許容誤差 (反復差の相対量に対する閾値)
)
DEFAULT_INITIAL_VALUE: float = (
    0.0001  # 内部格子点の初期値（0だと相対誤差で不安定な比が出るため小値）
)
DEFAULT_OUTPUT_DIR: str = "./output"  # 出力ディレクトリ
DEFAULT_PLOT_FILENAME: str = "2d_color_plot.png"  # プロット画像のファイル名
DEFAULT_RESULT_FILENAME: str = "calculated_result.txt"  # 結果ファイル名


# ------------------------------------------------------------
# データモデル: GridPoint（スカラー場の節点値）/ Solution（2D配列の集合）
# ------------------------------------------------------------


class GridPoint(BaseModel):
    """
    格子点の単一の値を表すデータモデル。
    Pydantic により型検証が行われる（例: value は float）。
    PDE的には、u(x_i,y_j) の近似値を保持するコンテナ。
    """

    value: float = Field(default=0.0, description="格子点の値（温度など）")


class Solution(BaseModel):
    """
    方程式の解（2次元格子全体のデータ）を表すモデル。
    2次元格子上の各点の値(GridPoint)をリストのリストとして保持し、
    値の取得/設定や外部ライブラリ連携（to_list）をカプセル化。
    """

    grid: list[list[GridPoint]] = Field(
        default_factory=list, description="2次元格子上のGridPointオブジェクトのリスト"
    )

    @classmethod
    def create_empty(cls, x_size: int, y_size: int) -> Self:
        """
        指定サイズの空の Solution を生成（全点 0 初期化）。
        Args:
            x_size: x方向の節点数（格子数+1）
            y_size: y方向の節点数（格子数+1）
        Returns:
            初期化済み Solution
        """
        # GridPoint(value=0.0) で2D配列を生成
        grid = [[GridPoint() for _ in range(y_size)] for _ in range(x_size)]
        return cls(grid=grid)

    def get_value(self, i: int, j: int) -> float:
        """
        節点 (i,j) の値 u_{i,j} を取得。
        """
        return self.grid[i][j].value

    def set_value(self, i: int, j: int, value: float) -> None:
        """
        節点 (i,j) に値を設定。
        """
        self.grid[i][j].value = value

    def to_list(self) -> list[list[float]]:
        """
        値のみの2次元リストへ変換（可視化・保存など外部連携用）。
        """
        return [[point.value for point in row] for row in self.grid]

    @classmethod
    def from_list(cls, data: list[list[float]]) -> Self:
        """
        値の2次元リストから Solution を生成（ディープコピー目的で使用）。
        """
        grid = [[GridPoint(value=val) for val in row] for row in data]
        return cls(grid=grid)


# ------------------------------------------------------------
# 設定データクラス: SimulationConfig
# ------------------------------------------------------------


@dataclass
class SimulationConfig:
    """
    シミュレーション設定を管理するデータクラス。
    PDEの文脈では (M,N) は分割数、h は格子幅。max_iterations は反復上限。
    error_tolerance は反復差の相対量に対する閾値（無次元）。
    """

    grid_counts_x: int  # M: x方向の格子数（内部点個数に近い概念。節点は0..M）
    grid_counts_y: int  # N: y方向の格子数
    grid_space: float  # h: 格子間隔
    max_iterations: int  # 反復上限（GSのスイープ回数）
    error_tolerance: float = DEFAULT_ERROR_TOLERANCE
    initial_value: float = DEFAULT_INITIAL_VALUE
    output_dir: str = DEFAULT_OUTPUT_DIR

    def __post_init__(self) -> None:
        """
        初期化後のバリデーション。
        数値解析的には M,N>=2 以上で内部点が存在することが重要（境界のみだと意味がない）。
        """
        if self.grid_counts_x <= 0 or self.grid_counts_y <= 0:
            raise ValueError("格子数は正の整数である必要があります")
        if self.grid_space <= 0:
            raise ValueError("格子間隔は正の値である必要があります")
        if self.max_iterations <= 0:
            raise ValueError("最大反復回数は正の整数である必要があります")
        if self.error_tolerance <= 0:
            raise ValueError("許容誤差は正の値である必要があります")
        # 出力ディレクトリ（冪等作成）
        os.makedirs(self.output_dir, exist_ok=True)


# ------------------------------------------------------------
# ソルバー本体: EllipticEquationSolver（ラプラス方程式, GS 法）
# ------------------------------------------------------------


class EllipticEquationSolver:
    """
    楕円型方程式（ここではラプラス方程式）をガウス–ザイデル法で解くクラス。
    内点の更新式は 5点平均 u_{i,j} ← (東+西+北+南)/4。
    Dirichlet 境界を固定したまま内部点を更新し続けることで、離散調和関数に収束。
    """

    def __init__(self, config: SimulationConfig):
        """
        Args:
            config: シミュレーション設定
        """
        self.config = config
        self.solution: Optional[Solution] = None  # 解（初期は None）
        self.iteration_count: int = 0  # 実行反復回数

    def initialize_arrays(self) -> None:
        """
        計算に使用する Solution を初期化。
        節点数は（格子数+1）なので x_points=M+1, y_points=N+1。
        """
        x_points = self.config.grid_counts_x + 1
        y_points = self.config.grid_counts_y + 1
        self.solution = Solution.create_empty(x_points, y_points)

    def set_initial_condition(self) -> None:
        """
        内部格子点（1..M-1, 1..N-1）の初期条件を設定。
        初期値を 0 ではなく小値にするのは、相対誤差（分母）が極小になるのを避けるための実務的工夫。
        """
        if self.solution is None:
            raise RuntimeError(
                "Solutionが初期化されていません。initialize_arraysを先に呼び出してください。"
            )
        for j in range(1, self.config.grid_counts_y):
            for i in range(1, self.config.grid_counts_x):
                self.solution.set_value(i, j, self.config.initial_value)

    def set_boundary_condition(self) -> None:
        """
        Dirichlet 境界条件を設定。
        下・左・右は u=0、上辺 y=Ly で u(x,Ly) = 4x(1-x) を課す。
        この境界設定により解は一意（離散最大値原理/DMP により内部極値は境界にのみ現れる）。
        """
        if self.solution is None:
            raise RuntimeError(
                "Solutionが初期化されていません。initialize_arraysを先に呼び出してください。"
            )

        # y=0（下辺）
        for i in range(self.config.grid_counts_x + 1):
            self.solution.set_value(i, 0, 0.0)

        # y=grid_counts_y（上辺）
        for i in range(self.config.grid_counts_x + 1):
            x = self.config.grid_space * i
            self.solution.set_value(i, self.config.grid_counts_y, 4.0 * x * (1.0 - x))

        # x=0（左辺）
        for j in range(1, self.config.grid_counts_y):
            self.solution.set_value(0, j, 0.0)

        # x=grid_counts_x（右辺）
        for j in range(1, self.config.grid_counts_y):
            self.solution.set_value(self.config.grid_counts_x, j, 0.0)

    def is_converged(self, previous_solution: Solution) -> bool:
        """
        収束判定（反復差の相対量）。
        注意: これは ||AU-b|| 残差ではないため、厳密性が必要な場合は残差ノルムの併用を推奨。
        （例）r_{i,j} = ((東+西+北+南) - 4u_{i,j})/h^2 の L∞ ノルムが tol 以下。
        """
        if self.solution is None:
            raise RuntimeError("Solutionが初期化されていません。")

        error_sum: float = 0.0  # |U^{k+1}-U^k| の総和
        value_sum: float = 0.0  # |U^{k+1}| の総和（相対化の分母）
        for j in range(1, self.config.grid_counts_y):
            for i in range(1, self.config.grid_counts_x):
                current_val = self.solution.get_value(i, j)
                previous_val = previous_solution.get_value(i, j)
                value_sum += abs(current_val)
                error_sum += abs(current_val - previous_val)

        # 分母が極小の場合の防御（初期段階の不安定な相対値を防ぐ）
        if value_sum < 1.0e-10:
            return True

        relative_error = error_sum / value_sum
        return relative_error <= self.config.error_tolerance

    def solve(self) -> tuple[Solution, int]:
        """
        ラプラス方程式（∇^2 u=0）をガウス–ザイデルで解く。
        流れ:
          1) Solution 初期化
          2) 初期条件設定（内部点）
          3) 境界条件設定（境界点）
          4) GS 反復: 内部点を (東+西+北+南)/4 で更新
          5) 反復差の相対量による停止判定

        Returns:
            (Solution, 反復回数)
        """
        self.initialize_arrays()
        if self.solution is None:
            raise RuntimeError("Solutionの初期化に失敗しました。")

        self.set_initial_condition()
        self.set_boundary_condition()
        self.iteration_count = 0

        for _ in range(self.config.max_iterations):
            # 収束判定用に前回解をディープコピー（値渡し）
            previous_solution = Solution.from_list(self.solution.to_list())

            # --- Gauss–Seidel 更新 ---
            # 5点ラプラシアンの右辺0 ⇒ 内点は近傍4点の単純平均に等しい（離散調和）
            for j in range(1, self.config.grid_counts_y):
                for i in range(1, self.config.grid_counts_x):
                    new_value = (
                        self.solution.get_value(i + 1, j)
                        + self.solution.get_value(i - 1, j)
                        + self.solution.get_value(i, j + 1)
                        + self.solution.get_value(i, j - 1)
                    ) / 4.0
                    # GS のため、更新は場に逐次反映（直後の点で最新値を使う）
                    self.solution.set_value(i, j, new_value)

            self.iteration_count += 1

            # --- 停止判定（反復差） ---
            if self.is_converged(previous_solution):
                print(f"収束しました（反復回数: {self.iteration_count}）")
                break
        else:
            # 反復上限に達した（GS は遅いことがある。SOR/MG 導入で改善可能）
            print(
                f"警告: 最大反復回数（{self.config.max_iterations}）に達しました。\n"
                f"収束していない可能性があります。"
            )

        if self.solution is None:
            raise RuntimeError("予期せぬエラー: SolutionがNoneです。")
        return self.solution, self.iteration_count


# ------------------------------------------------------------
# 可視化/出力: ResultVisualizer
# ------------------------------------------------------------


class ResultVisualizer:
    """
    計算結果の可視化とファイル出力を担うクラス（関心分離）。
    理論補足: imshow の extent を [-h/2, L+h/2] にとると、セル中心と座標目盛の整合が視覚的に良い。
    """

    def __init__(self, config: SimulationConfig):
        """
        Args:
            config: 出力先ディレクトリ等の情報を利用
        """
        self.config = config

    def create_color_plot(self, solution: Solution) -> None:
        """
        計算結果を 2D カラーマップで保存。
        """
        filename = os.path.join(self.config.output_dir, DEFAULT_PLOT_FILENAME)

        # 軸範囲（セル中心が [0, L] に並ぶよう ±h/2 だけ広げる）
        min_x_y = -self.config.grid_space / 2
        max_x_y = (
            self.config.grid_space * self.config.grid_counts_x
            + self.config.grid_space / 2
        )
        extent = (min_x_y, max_x_y, min_x_y, max_x_y)

        # Solution → ndarray にして転置（imshow は [行(y), 列(x)]）
        solution_array = np.array(solution.to_list()).T

        plt.figure(figsize=(10, 8))
        plt.title("2D Temperature Distribution (Object-Oriented)")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        img = plt.imshow(
            solution_array,
            cmap="viridis",
            interpolation="none",
            aspect="auto",
            origin="lower",
            extent=extent,
        )
        plt.colorbar(img, label="Temperature")
        plt.xticks(
            np.arange(
                0,
                self.config.grid_space * (self.config.grid_counts_x + 1),
                self.config.grid_space * 2,
            )
        )
        plt.yticks(
            np.arange(
                0,
                self.config.grid_space * (self.config.grid_counts_y + 1),
                self.config.grid_space * 2,
            )
        )
        plt.savefig(filename, format="png", dpi=300)
        plt.close()
        print(f"カラープロットを保存しました: {filename}")

    def save_result_to_file(
        self,
        solution: Solution,
        iteration_count: int,
    ) -> None:
        """
        計算結果と設定パラメータをテキスト出力。
        """
        filename = os.path.join(self.config.output_dir, DEFAULT_RESULT_FILENAME)
        with open(filename, "w", encoding="utf-8") as file:
            file.write(
                "# Elliptic Equation Solution (Gauss-Seidel Method - OOP Version)\n\n"
            )
            file.write("# Calculation Parameters\n")
            file.write(f"#   grid_counts_x: {self.config.grid_counts_x}\n")
            file.write(f"#   grid_counts_y: {self.config.grid_counts_y}\n")
            file.write(f"#   grid_space: {self.config.grid_space}\n")
            file.write(f"#   error_tolerance: {self.config.error_tolerance}\n")
            file.write(f"#   initial_value: {self.config.initial_value}\n")
            file.write(f"#   max_iterations: {self.config.max_iterations}\n")
            file.write(f"#   actual_iterations: {iteration_count}\n\n")
            file.write("# Solution Matrix (Temperature Distribution)\n")
            for row in solution.to_list():
                line = " ".join(map(str, row))
                file.write(line + "\n")
        print(f"計算結果をファイルに保存しました: {filename}")


# ------------------------------------------------------------
# 実行フロー: run_simulation / main
# ------------------------------------------------------------


def run_simulation(config: SimulationConfig) -> None:
    """
    シミュレーション全体の実行フロー:
      1) ソルバーで計算
      2) 可視化
      3) 結果ファイル保存
    """
    print("シミュレーションを開始します...")
    solver = EllipticEquationSolver(config)
    solution, iteration_count = solver.solve()

    visualizer = ResultVisualizer(config)
    visualizer.create_color_plot(solution)
    visualizer.save_result_to_file(solution, iteration_count)
    print("シミュレーションが完了しました。")


def main() -> None:
    """メイン関数: プログラムのエントリーポイント"""
    try:
        # シミュレーション設定
        config = SimulationConfig(
            grid_counts_x=10,  # M: x方向の格子数
            grid_counts_y=10,  # N: y方向の格子数
            grid_space=0.1,  # h: 格子間隔（総長さ L≈Mh）
            max_iterations=1000,  # 反復上限
            # error_tolerance, initial_value, output_dir はデフォルト利用
        )
        run_simulation(config)
    except ValueError as ve:
        print(f"設定エラー: {ve}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        # 追跡が必要なら以下を有効化
        # import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()  # スクリプトとして実行された場合に main() を呼び出す

# ============================================================
# 参考メモ（実装には影響しない理論的追記）
# ------------------------------------------------------------
# • 残差停止の併用（推奨）:
#     r_{i,j} = ((東+西+北+南) - 4u_{i,j})/h^2 の L∞ ノルムが tol 以下 を停止条件に追加。
#   反復差だけだと「ほぼ変化しないが PDE として不十分」な場合を見逃す可能性がある。
#
# • 高速化:
#     SOR（緩和）: u ← (1-ω)u + ω*平均, ω≈2/(1+sin(pi/(max(M,N)+1))) を初期値に探索。
#     マルチグリッド: 低周波誤差を粗格子で処理し格子独立に近い収束。
#
# • 検証:
#     解析参照解（上辺 f(x)=4x(1-x) の正弦展開）と比較、または二/三段格子でオーダー検証（≈2）。
# ============================================================
