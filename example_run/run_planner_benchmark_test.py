

__all__ = ['demo', 'teaser']

def teaser():
    return demo() # 执行spider.teaser(), 就会执行demo()

def demo():
    from gops.utils.planner_benchmark.interface.BaseBenchmark import DummyBenchmark
    from gops.utils.planner_benchmark.planner_zoo import LatticePlanner, BezierPlanner, IDMPlanner

    planner = LatticePlanner({
        "steps": 20,
        "dt": 0.2,
        "end_s_candidates": (20, 40, 60),
        "end_l_candidates": (-3.5, 0, 3.5),
    })

    # planner = BezierPlanner({
    #     "steps": 20,
    #     "dt": 0.2,
    #     "end_s_candidates": (20,30, 60),
    #     "end_l_candidates": (-3.5, 0, 3.5),
    #     "end_v_candidates": tuple(i * 60 / 3.6 / 3 for i in range(4)),  # 改这一项的时候，要连着限速一起改了
    #     "end_T_candidates": (2, 4, 8),  # s_dot, T采样生成纵向轨迹
    # })
    # planner = IDMPlanner()
    benchmark = DummyBenchmark()
    benchmark.test(planner)



if __name__ == '__main__':
    demo()

