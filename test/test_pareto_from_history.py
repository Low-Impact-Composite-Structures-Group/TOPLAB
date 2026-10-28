from toplab.plotting.optimization_plots import pareto_from_history


def _row(r, a, b, feasible=True, error=""):
    return {"evaluation": 1, "radius_1": r, "eta_g": a, "eta_v": b, "feasible": feasible, "error": error}


def test_non_dominated_feasible_only_and_deduplicated():
    history = [
        _row(1, 0.9, 0.5),
        _row(1, 0.9, 0.5),  # duplicate design (parallel ranks writing the same row)
        _row(2, 0.5, 0.9),
        _row(3, 0.4, 0.4),  # dominated
        _row(4, 1.0, 1.0, feasible=False),  # would dominate everything but is infeasible
        _row(5, 1.0, 1.0, error="boom"),
    ]
    front = pareto_from_history(history, ["eta_g", "eta_v"])
    assert sorted(r["radius_1"] for r in front) == [1, 2]


def test_falls_back_to_all_rows_when_nothing_feasible():
    history = [_row(1, 0.9, 0.5, feasible=False), _row(2, 0.4, 0.4, feasible=False)]
    assert [r["radius_1"] for r in pareto_from_history(history, ["eta_g", "eta_v"])] == [1]
