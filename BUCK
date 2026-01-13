load("@fbcode_macros//build_defs:python_library.bzl", "python_library")

oncall("fill in oncall here @nocommit")

python_library(
    name = "covariate_shift",
    srcs = ["covariate_shift.py"],
    labels = ["autodeps2_generated"],
    deps = [
        "fbsource//third-party/pypi/cvxopt:cvxopt",
        "fbsource//third-party/pypi/numpy:numpy",
    ],
)

python_library(
    name = "quantile_match",
    srcs = ["quantile_match.py"],
    labels = ["autodeps2_generated"],
    deps = [
        "fbsource//third-party/pypi/numpy:numpy",
        "fbsource//third-party/pypi/scikit-learn:scikit-learn",
        "fbsource//third-party/pypi/scipy:scipy",
    ],
)
