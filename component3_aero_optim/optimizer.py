"""
Driving-cycle-aware shape optimiser.

Searches the 5 shape parameters, within realistic styling limits, for the
combination that minimises energy consumed over a chosen driving cycle —
NOT the lowest drag coefficient in the abstract. Uses Optuna's default TPE
(Tree-structured Parzen Estimator) sampler, a Bayesian-optimisation method,
because the deployed drag surrogate may be a non-differentiable tree model
(XGBoost/Random Forest), ruling out gradient descent. Optuna's CmaEsSampler
is available as the spec's named CMA-ES fallback via a one-line sampler
swap, without a second dependency.

Each trial: shape params -> surrogate -> Cd, and (if an area model is given)
shape params -> frontal area A -> energy_model with drag area Cd*A -> cycle
energy (Wh/km). Without an area model the constant config.FRONTAL_AREA_M2
is used. Tracks the best shape found over the whole search (not the most
recently tried one — Optuna's study.best_trial already does this natively).
Compares the best-found shape against the baseline with a tolerance band:
within it, report "already near-optimal"; beyond it, report the better
shape found plus the resulting range/cost improvement.
"""

import optuna

import config
from energy_model import load_driving_cycle, prepare_cycle, simulate_cycle
from surrogate_training import predict_cd, predict_frontal_area

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _evaluate_shape(model, area_model, cycle, shape, vehicle_kwargs):
    """One shape -> drag, frontal area (if modelled), energy result."""
    cd = predict_cd(model, shape)
    kwargs = dict(vehicle_kwargs)
    if area_model is not None:
        if "frontal_area_m2" in kwargs:
            raise ValueError("Give either area_model or frontal_area_m2 in vehicle_kwargs, not both.")
        kwargs["frontal_area_m2"] = predict_frontal_area(area_model, shape)
    result = simulate_cycle(cycle, cd, return_trace=False, **kwargs)
    result["frontal_area_m2"] = kwargs.get("frontal_area_m2", config.FRONTAL_AREA_M2)
    result["cd_a_m2"] = cd * result["frontal_area_m2"]
    return result


def _objective_factory(model, area_model, cycle_df, vehicle_kwargs):
    def objective(trial):
        params = {
            name: trial.suggest_float(name, *config.SHAPE_PARAM_BOUNDS[name])
            for name in config.SHAPE_PARAM_NAMES
        }
        return _evaluate_shape(model, area_model, cycle_df, params, vehicle_kwargs)["wh_per_km"]

    return objective


def optimise_shape(
    model,
    cycle_name,
    n_trials=config.OPTUNA_N_TRIALS,
    sampler="tpe",
    vehicle_kwargs=None,
    enqueue_shape=None,
    area_model=None,
):
    """
    Run the optimiser for one driving cycle. Returns a dict with the best
    shape found, its Wh/km, and the Optuna study (for convergence plots).

    If enqueue_shape is given (e.g. the baseline shape), it is evaluated as
    the very first trial via study.enqueue_trial — this guarantees the
    search's "best found" can never be worse than that shape, rather than
    relying on the sampler happening to explore something at least as good
    within n_trials by chance.
    """
    vehicle_kwargs = vehicle_kwargs or {}
    cycle_df = prepare_cycle(load_driving_cycle(cycle_name))

    if sampler == "cmaes":
        optuna_sampler = optuna.samplers.CmaEsSampler(seed=config.RANDOM_STATE)
    else:
        optuna_sampler = optuna.samplers.TPESampler(seed=config.RANDOM_STATE)

    study = optuna.create_study(direction="minimize", sampler=optuna_sampler)
    if enqueue_shape is not None:
        study.enqueue_trial(enqueue_shape)
    study.optimize(_objective_factory(model, area_model, cycle_df, vehicle_kwargs), n_trials=n_trials)

    best_shape = study.best_trial.params
    best_wh_per_km = study.best_value
    return {"shape": best_shape, "wh_per_km": best_wh_per_km, "study": study}


def compare_to_baseline(
    model,
    cycle_name,
    baseline_shape=config.BASELINE_SHAPE,
    tolerance=config.BASELINE_OPTIMAL_TOLERANCE,
    n_trials=config.OPTUNA_N_TRIALS,
    vehicle_kwargs=None,
    area_model=None,
):
    """
    The check-then-search decision described in the spec: evaluate the
    baseline shape, run the optimiser, and decide whether the baseline is
    already near-optimal (within `tolerance`, relative) or a better shape
    was found — reporting the resulting range and cost change either way.
    """
    vehicle_kwargs = vehicle_kwargs or {}
    cycle_df = prepare_cycle(load_driving_cycle(cycle_name))

    baseline_result = _evaluate_shape(model, area_model, cycle_df, baseline_shape, vehicle_kwargs)

    optimisation = optimise_shape(
        model,
        cycle_name,
        n_trials=n_trials,
        vehicle_kwargs=vehicle_kwargs,
        enqueue_shape=baseline_shape,
        area_model=area_model,
    )
    best_shape = optimisation["shape"]
    best_result = _evaluate_shape(model, area_model, cycle_df, best_shape, vehicle_kwargs)

    relative_improvement = (
        (baseline_result["wh_per_km"] - best_result["wh_per_km"]) / baseline_result["wh_per_km"]
        if baseline_result["wh_per_km"] > 0
        else 0.0
    )

    already_near_optimal = bool(relative_improvement <= tolerance)

    delta_range_km = best_result["range_km"] - baseline_result["range_km"]
    delta_cost_lkr = baseline_result["cost_lkr"] - best_result["cost_lkr"]

    return {
        "cycle": cycle_name,
        "already_near_optimal": already_near_optimal,
        "relative_improvement": relative_improvement,
        "baseline": {"shape": baseline_shape, **baseline_result},
        "optimised": {"shape": best_shape, **best_result},
        "delta_range_km": delta_range_km,
        "delta_cost_lkr": delta_cost_lkr,
    }
