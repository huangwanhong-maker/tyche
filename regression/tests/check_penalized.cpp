// Checks of tyche/regression/penalized.hpp: ridge, the lasso and the elastic net along a path, for the
// gaussian and binomial families, and K-fold cross-validation.
//
// - Closed forms: on an orthonormal design (centred columns, XᵀX/N = I) the lasso and elastic net are
//   S(xⱼᵀy/N, λα)/(1 + λ(1 − α)) (Friedman, Hastie and Tibshirani 2010, eq. 5); ridge in closed form
//   equals the normal equations' solution and the coordinate descent at α = 0; at λ = 0 the lasso is
//   least squares (tyche's ols) and the binomial lasso is logistic regression (tyche's glm).
// - λ_max: every penalised coefficient 0 there, and not just below it.
// - The Karush–Kuhn–Tucker conditions along whole paths, gaussian and binomial, standardised or not,
//   weighted, with penalty factors (one column unpenalised), and with p > N.
// - scikit-learn 1.9.1 (cross-check, fixtures/penalized-reference.txt): Lasso and ElasticNet (with and
//   without weights, on X and on X standardised there), LogisticRegression (saga, elastic-net penalty).
// - Cross-validation: balanced deterministic folds; a fold's error recomputed by hand; λ_1se ≥ λ_min;
//   a sparse truth recovered at λ_1se.

#include <tyche/regression/glm.hpp>
#include <tyche/regression/linear.hpp>
#include <tyche/regression/penalized.hpp>

#include <calx/linalg/dense.hpp>
#include <calx/linalg/qr.hpp>
#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace rg = tyche::regression;
using calx::linalg::Rectangular;

namespace {

Rectangular random_design(calx::random::Stream& z, std::size_t n, std::size_t p, double rho = 0.4) {
  Rectangular X(n, p);
  for (std::size_t i = 0; i < n; i++) {
    double prev = z.normal();
    for (std::size_t j = 0; j < p; j++) {
      const double v = j == 0 ? prev : rho * prev + std::sqrt(1 - rho * rho) * z.normal();
      X(i, j) = (1 + j % 3) * v + 0.5 * static_cast<double>(j);
      prev = v;
    }
  }
  return X;
}

double soft(double z, double t) { return z > t ? z - t : z < -t ? z + t : 0.0; }

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0;
  for (std::size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
  return a.size() == b.size() ? m : INFINITY;
}

}  // namespace

int main() {
  calx::verification::Suite s;

  // ---- the orthonormal design ----
  {
    calx::random::Stream z(1, 0, 0, 0);
    const std::size_t n = 100, p = 6;
    Rectangular G(n, p);
    for (auto& v : G.a) v = z.normal();
    for (std::size_t j = 0; j < p; j++) {
      double m = 0;
      for (std::size_t i = 0; i < n; i++) m += G(i, j) / n;
      for (std::size_t i = 0; i < n; i++) G(i, j) -= m;
    }
    const auto Q = calx::linalg::thin_q(calx::linalg::qr(G, false));
    Rectangular X(n, p);
    for (std::size_t i = 0; i < n; i++)
      for (std::size_t j = 0; j < p; j++) X(i, j) = std::sqrt(static_cast<double>(n)) * Q(i, j);
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; i++) y[i] = 2 + 1.5 * X(i, 0) - 0.4 * X(i, 1) + 0.1 * X(i, 2) + 0.7 * z.normal();
    double worst = 0;
    for (const double alpha : {1.0, 0.6, 0.2}) {
      for (const double lambda : {0.05, 0.3, 1.0}) {
        rg::PenalizedOptions o;
        o.alpha = alpha;
        o.standardize = false;
        const auto f = rg::elastic_net(X, y, lambda, o);
        double ybar = 0;
        for (const double v : y) ybar += v / n;
        for (std::size_t j = 0; j < p; j++) {
          double g = 0;
          for (std::size_t i = 0; i < n; i++) g += X(i, j) * y[i] / n;
          worst = std::max(worst, std::fabs(f.beta[j] - soft(g, lambda * alpha) / (1 + lambda * (1 - alpha))));
        }
        worst = std::max(worst, std::fabs(f.intercept - ybar));
      }
    }
    s.check("orthonormal design: β = S(xⱼᵀy/N, λα)/(1 + λ(1 − α)) and β₀ = ȳ (Friedman et al. 2010, eq. 5)", worst, 1e-9);
  }

  // ---- ridge in closed form ----
  {
    calx::random::Stream z(2, 0, 0, 0);
    const std::size_t n = 80, p = 7;
    const auto X = random_design(z, n, p);
    std::vector<double> y(n), w(n);
    for (std::size_t i = 0; i < n; i++) {
      y[i] = 1 + 0.5 * X(i, 0) - 0.3 * X(i, 3) + z.normal();
      w[i] = 0.5 + z.uniform();
    }
    double worst = 0, worst_cd = 0;
    for (const bool standardize : {false, true}) {
      rg::PenalizedOptions o;
      o.standardize = standardize;
      o.weights = w;
      const double lambda = 0.7;
      const auto r = rg::ridge(X, y, lambda, o);
      // The normal equations on the centred (and scaled) columns with v = w/Σw.
      double sw = 0;
      for (const double v : w) sw += v;
      std::vector<double> c(p, 0.0), sd(p, 1.0);
      for (std::size_t j = 0; j < p; j++) {
        for (std::size_t i = 0; i < n; i++) c[j] += w[i] / sw * X(i, j);
        double ss = 0;
        for (std::size_t i = 0; i < n; i++) ss += w[i] / sw * (X(i, j) - c[j]) * (X(i, j) - c[j]);
        if (standardize) sd[j] = std::sqrt(ss);
      }
      double yb = 0;
      for (std::size_t i = 0; i < n; i++) yb += w[i] / sw * y[i];
      calx::linalg::Matrix A(p), B(p);
      for (std::size_t a = 0; a < p; a++) {
        for (std::size_t b = 0; b < p; b++) {
          double v = 0;
          for (std::size_t i = 0; i < n; i++) v += w[i] / sw * (X(i, a) - c[a]) / sd[a] * (X(i, b) - c[b]) / sd[b];
          A(a, b) = v + (a == b ? lambda : 0.0);
        }
        double v = 0;
        for (std::size_t i = 0; i < n; i++) v += w[i] / sw * (X(i, a) - c[a]) / sd[a] * (y[i] - yb);
        B(a, 0) = v;
      }
      const auto sol = calx::linalg::solve(A, B);
      for (std::size_t j = 0; j < p; j++) worst = std::max(worst, std::fabs(r.beta[j] - sol(j, 0) / sd[j]));
      rg::PenalizedOptions q = o;
      q.alpha = 0;
      const auto cd = rg::elastic_net(X, y, lambda, q);
      worst_cd = std::max({worst_cd, max_abs_diff(cd.beta, r.beta), std::fabs(cd.intercept - r.intercept)});
    }
    s.check("ridge: the closed form (QR of the augmented design) equals the normal equations' solution, weighted, standardised or not", worst, 1e-10);
    s.check("ridge: coordinate descent at α = 0 equals the closed form", worst_cd, 1e-7);
  }

  // ---- λ = 0: least squares and logistic regression ----
  {
    calx::random::Stream z(3, 0, 0, 0);
    const std::size_t n = 150, p = 5;
    const auto X = random_design(z, n, p);
    std::vector<double> y(n), yb(n);
    for (std::size_t i = 0; i < n; i++) {
      const double eta = 0.3 + 0.4 * X(i, 0) - 0.2 * X(i, 2) + 0.1 * X(i, 4);
      y[i] = eta + z.normal();
      yb[i] = z.uniform() < 1 / (1 + std::exp(-eta)) ? 1.0 : 0.0;
    }
    std::vector<std::vector<double>> cols;
    for (std::size_t j = 0; j < p; j++) cols.push_back(X.column(j));
    const Rectangular D = rg::design_matrix(cols, true);
    const auto ols = rg::ols(D, y);
    const auto lasso0 = rg::lasso(X, y, 0.0);
    double d = std::fabs(lasso0.intercept - ols.beta[0]);
    for (std::size_t j = 0; j < p; j++) d = std::max(d, std::fabs(lasso0.beta[j] - ols.beta[j + 1]));
    s.check("λ = 0: the lasso is least squares (tyche's ols)", d, 1e-8);
    rg::GlmOptions go;
    go.family = rg::Family::binomial;
    const auto lr = rg::glm(D, yb, go);
    rg::PenalizedOptions po;
    po.family = rg::PenalizedFamily::binomial;
    const auto blasso0 = rg::lasso(X, yb, 0.0, po);
    double e = std::fabs(blasso0.intercept - lr.beta[0]);
    for (std::size_t j = 0; j < p; j++) e = std::max(e, std::fabs(blasso0.beta[j] - lr.beta[j + 1]));
    s.check("λ = 0: the binomial lasso is logistic regression (tyche's glm)", e, 1e-6);
    s.check("λ = 0: the binomial deviance equals the GLM's", blasso0.deviance - lr.deviance, 1e-7);
  }

  // ---- λ_max and the KKT conditions along paths ----
  {
    calx::random::Stream z(4, 0, 0, 0);
    double worst = 0;
    std::size_t paths = 0;
    bool lmax_ok = true;
    std::string where;
    for (int rep = 0; rep < 12; rep++) {
      const std::size_t n = rep % 4 == 3 ? 40 : 120, p = rep % 4 == 3 ? 70 : 9;
      const auto X = random_design(z, n, p, 0.6);
      std::vector<double> y(n), w(n);
      for (std::size_t i = 0; i < n; i++) {
        const double eta = -0.2 + 0.6 * X(i, 0) - 0.5 * X(i, 1) + 0.3 * X(i, 5);
        y[i] = rep % 2 ? (z.uniform() < 1 / (1 + std::exp(-eta)) ? 1.0 : 0.0) : eta + z.normal();
        w[i] = std::exp(z.normal(0, 0.4));
      }
      rg::PenalizedOptions o;
      o.family = rep % 2 ? rg::PenalizedFamily::binomial : rg::PenalizedFamily::gaussian;
      o.alpha = rep % 3 == 0 ? 1.0 : rep % 3 == 1 ? 0.5 : 0.05;
      o.standardize = rep % 4 != 1;
      if (rep % 4 == 2) o.weights = w;
      if (rep % 6 == 5) {
        o.penalty_factor.assign(p, 1.0);
        o.penalty_factor[1] = 0;
      }
      o.lambda_count = 40;
      const auto path = rg::elastic_net_path(X, y, o);
      paths++;
      for (const auto& f : path.fits) {
        const double k = rg::penalized_kkt(X, y, f, o);
        if (k > worst) {
          worst = k;
          where = "path " + std::to_string(rep) + ", λ = " + std::to_string(f.lambda);
        }
      }
      // At λ_max every penalised β is 0; just below, one is not (α ≥ 10⁻³).
      bool zero = true;
      for (std::size_t j = 0; j < p; j++)
        if ((o.penalty_factor.empty() || o.penalty_factor[j] > 0) && path.fits[0].beta[j] != 0) zero = false;
      const auto below = rg::elastic_net(X, y, path.lambda_max * 0.999, o);
      std::size_t nz = 0;
      for (std::size_t j = 0; j < p; j++)
        if ((o.penalty_factor.empty() || o.penalty_factor[j] > 0) && below.beta[j] != 0) nz++;
      if (!zero || nz == 0) lmax_ok = false;
    }
    s.check("KKT conditions along " + std::to_string(paths) + " paths (gaussian and binomial, α 1, 0.5, 0.05, weights, an unpenalised column, p > N), relative to λ (" + where + ")",
            worst, 1e-5);
    s.require("λ_max: every penalised coefficient 0 at λ_max, and some not at 0.999 λ_max", lmax_ok);
  }

  // ---- scikit-learn (cross-check) ----
  {
    std::ifstream in(std::string(TYCHE_FIXTURES) + "/penalized-reference.txt");
    s.require("scikit-learn fixtures: the reference file opens", static_cast<bool>(in));
    struct Data {
      Rectangular X;
      std::vector<double> y, w;
    };
    std::map<int, Data> data;
    std::string line;
    double worst_g = 0, worst_b = 0;
    std::size_t fits_g = 0, fits_b = 0;
    auto numbers = [](const std::string& l) {
      std::vector<double> v;
      std::istringstream is(l);
      for (double x; is >> x;) v.push_back(x);
      return v;
    };
    while (std::getline(in, line)) {
      if (line.rfind("data ", 0) == 0) {
        std::istringstream h(line.substr(5));
        int id;
        std::size_t n, p;
        h >> id >> n >> p;
        Data d;
        d.X = Rectangular(n, p);
        std::getline(in, line);
        d.X.a = numbers(line);
        std::getline(in, line);
        d.y = numbers(line);
        std::getline(in, line);
        d.w = numbers(line);
        data[id] = std::move(d);
      } else if (line.rfind("fit ", 0) == 0) {
        std::istringstream h(line.substr(4));
        int id, standardize, weighted;
        std::string fam;
        double alpha, lambda;
        h >> id >> fam >> alpha >> lambda >> standardize >> weighted;
        std::getline(in, line);
        const auto ref = numbers(line);
        const Data& d = data.at(id);
        rg::PenalizedOptions o;
        o.family = fam == "binomial" ? rg::PenalizedFamily::binomial : rg::PenalizedFamily::gaussian;
        o.alpha = alpha;
        o.standardize = standardize != 0;
        if (weighted) o.weights = d.w;
        const auto f = rg::elastic_net(d.X, d.y, lambda, o);
        double diff = std::fabs(f.intercept - ref[0]);
        for (std::size_t j = 0; j < f.beta.size(); j++) diff = std::max(diff, std::fabs(f.beta[j] - ref[j + 1]) * std::max(1.0, std::sqrt(d.X.rows) * 0.0 + 1.0));
        if (fam == "binomial") {
          worst_b = std::max(worst_b, diff);
          fits_b++;
        } else {
          worst_g = std::max(worst_g, diff);
          fits_g++;
        }
      }
    }
    s.check("scikit-learn 1.9.1 Lasso and ElasticNet: " + std::to_string(fits_g) + " fits (weights, standardised or not, p > N), the largest coefficient difference", worst_g, 1e-7);
    s.check("scikit-learn 1.9.1 LogisticRegression (saga, elastic net): " + std::to_string(fits_b) + " fits, the largest coefficient difference", worst_b, 1e-5);
  }

  // ---- cross-validation ----
  {
    calx::random::Stream z(5, 0, 0, 0);
    const std::size_t n = 200, p = 40;
    const auto X = random_design(z, n, p, 0.3);
    const std::vector<std::size_t> truth{0, 3, 7, 12, 20};
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; i++) {
      double eta = 1;
      for (const std::size_t j : truth) eta += 0.8 * X(i, j) / (1 + j % 3);
      y[i] = eta + z.normal();
    }
    rg::PenalizedOptions o;
    o.lambda_count = 50;
    rg::CvOptions cv;
    cv.folds = 10;
    const auto c = rg::cross_validate(X, y, o, cv);
    std::vector<std::size_t> count(10, 0);
    for (const std::size_t f : c.fold) count[f]++;
    s.require("cross-validation: ten balanced folds of 20", std::all_of(count.begin(), count.end(), [](std::size_t k) { return k == 20; }));
    const auto c2 = rg::cross_validate(X, y, o, cv);
    s.require("cross-validation: the same seed gives the same folds and errors", c.fold == c2.fold && c.mean_error == c2.mean_error);
    s.require("cross-validation: λ_1se ≥ λ_min, and cvm(λ_1se) ≤ cvm(λ_min) + cvsd(λ_min)", c.lambda_1se >= c.lambda_min &&
              c.mean_error[c.index_1se] <= c.mean_error[c.index_min] + c.standard_error[c.index_min]);
    // Fold 3's error at λ_min, recomputed by refitting by hand.
    std::vector<std::size_t> train, test;
    for (std::size_t i = 0; i < n; i++) (c.fold[i] == 3 ? test : train).push_back(i);
    Rectangular Xt(train.size(), p);
    std::vector<double> yt(train.size());
    for (std::size_t r = 0; r < train.size(); r++) {
      for (std::size_t j = 0; j < p; j++) Xt(r, j) = X(train[r], j);
      yt[r] = y[train[r]];
    }
    rg::PenalizedOptions q = o;
    q.lambda = c.path.lambda;
    const auto fp = rg::elastic_net_path(Xt, yt, q);
    const auto& f = fp.fits[c.index_min];
    double e3 = 0;
    for (const std::size_t i : test) {
      double eta = f.intercept;
      for (std::size_t j = 0; j < p; j++) eta += X(i, j) * f.beta[j];
      e3 += (y[i] - eta) * (y[i] - eta) / static_cast<double>(test.size());
    }
    // cvm is the weighted mean of the ten folds' errors; recompute all ten to compare.
    double all = 0;
    for (std::size_t k = 0; k < 10; k++) {
      std::vector<std::size_t> tr, te;
      for (std::size_t i = 0; i < n; i++) (c.fold[i] == k ? te : tr).push_back(i);
      Rectangular Xk(tr.size(), p);
      std::vector<double> yk(tr.size());
      for (std::size_t r = 0; r < tr.size(); r++) {
        for (std::size_t j = 0; j < p; j++) Xk(r, j) = X(tr[r], j);
        yk[r] = y[tr[r]];
      }
      const auto fk = rg::elastic_net_path(Xk, yk, q).fits[c.index_min];
      for (const std::size_t i : te) {
        double eta = fk.intercept;
        for (std::size_t j = 0; j < p; j++) eta += X(i, j) * fk.beta[j];
        all += (y[i] - eta) * (y[i] - eta) / static_cast<double>(n);
      }
    }
    s.check("cross-validation: cvm(λ_min) recomputed from ten refits by hand", c.mean_error[c.index_min] - all, 1e-10);
    s.require("cross-validation: a fold's own error is finite and positive", e3 > 0 && std::isfinite(e3));
    const auto& best = c.path.fits[c.index_1se];
    std::size_t found = 0, extra = 0;
    for (std::size_t j = 0; j < p; j++) {
      const bool in_truth = std::find(truth.begin(), truth.end(), j) != truth.end();
      if (best.beta[j] != 0) (in_truth ? found : extra)++;
    }
    s.require("cross-validation: at λ_1se the five true columns are selected, with at most five others", found == 5 && extra <= 5,
              std::to_string(found) + " true, " + std::to_string(extra) + " others");
    // Binomial cross-validation by deviance and misclassification.
    std::vector<double> yb(n);
    for (std::size_t i = 0; i < n; i++) {
      double eta = -0.5;
      for (const std::size_t j : truth) eta += 0.9 * X(i, j) / (1 + j % 3);
      yb[i] = z.uniform() < 1 / (1 + std::exp(-eta)) ? 1.0 : 0.0;
    }
    rg::PenalizedOptions ob = o;
    ob.family = rg::PenalizedFamily::binomial;
    ob.alpha = 0.5;
    const auto cb = rg::cross_validate(X, yb, ob, cv);
    rg::CvOptions cm = cv;
    cm.measure = rg::CvMeasure::misclassification;
    const auto cbm = rg::cross_validate(X, yb, ob, cm);
    s.require("cross-validation (binomial): deviance and misclassification measured, λ_1se ≥ λ_min", cb.measure == rg::CvMeasure::deviance && cb.lambda_1se >= cb.lambda_min &&
              cbm.mean_error[cbm.index_min] < 0.5);
    const auto pr = rg::predict(cb.path.fits[cb.index_min], X);
    s.require("predict (binomial): probabilities in [0, 1]", std::all_of(pr.begin(), pr.end(), [](double v) { return v >= 0 && v <= 1; }));
  }
  return s.done();
}
