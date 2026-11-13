// Checks of tyche/compositions/simplex and balances: Aitchison's geometry and its orthonormal coordinates.
//
// Closure: every operation invariant to the scale of its arguments and to the total asked. The clr sums to
// zero and is inverted. Perturbation is a commutative group (neutral, inverse, associativity) and powering
// distributes over it, so that the simplex is a vector space; the inner product's two forms agree, the
// distance is invariant under perturbation and scales under powering. Balances: the contrast matrix of
// every partition tried (pivot coordinates of 2 to 8 parts in every rotation, hand-written partitions,
// random partitions) is orthonormal with rows summing to zero; ilr is an isometry and a linear
// isomorphism, inverted; a balance equals its coordinate; pivot coordinates match Hron et al.'s formula;
// a change of basis is orthogonal. Invalid partitions are refused. Cross-checks against scikit-bio's
// composition module (tests/fixtures/make_skbio_composition_fixtures.py).

#include <tyche/compositions/balances.hpp>
#include <tyche/compositions/simplex.hpp>

#include <calx/random/stream.hpp>
#include <calx/verification/check.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace tyche::compositions;

namespace {

template <class F>
bool throws(F f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

double gap(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return INFINITY;
  double m = 0;
  for (std::size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

std::vector<double> random_composition(calx::random::Stream& rng, std::size_t D, double spread = 1.5) {
  std::vector<double> x(D);
  for (double& v : x) v = std::exp(spread * rng.normal()) * 37.0;
  return x;
}

// A random sequential binary partition of D parts: split the whole, then each group, at random.
SignTable random_partition(calx::random::Stream& rng, std::size_t D) {
  SignTable rows;
  std::vector<std::vector<std::size_t>> queue;
  std::vector<std::size_t> all(D);
  std::iota(all.begin(), all.end(), std::size_t{0});
  queue.push_back(all);
  while (!queue.empty()) {
    auto g = queue.front();
    queue.erase(queue.begin());
    for (std::size_t i = g.size() - 1; i > 0; i--) std::swap(g[i], g[rng.below(i + 1)]);
    const std::size_t cut = 1 + rng.below(g.size() - 1);
    std::vector<std::size_t> plus(g.begin(), g.begin() + static_cast<std::ptrdiff_t>(cut)), minus(g.begin() + static_cast<std::ptrdiff_t>(cut), g.end());
    std::sort(plus.begin(), plus.end());
    std::sort(minus.begin(), minus.end());
    std::vector<int> row(D, 0);
    for (std::size_t i : plus) row[i] = 1;
    for (std::size_t i : minus) row[i] = -1;
    rows.push_back(row);
    if (plus.size() >= 2) queue.push_back(plus);
    if (minus.size() >= 2) queue.push_back(minus);
  }
  return rows;
}

// max |ΨΨᵀ − I| and max |Ψ1|.
std::pair<double, double> orthonormality(const Basis& b) {
  double o = 0, s = 0;
  for (std::size_t k = 0; k < b.contrast.rows; k++) {
    double row = 0;
    for (std::size_t j = 0; j < b.parts; j++) row += b.contrast(k, j);
    s = std::max(s, std::fabs(row));
    for (std::size_t l = 0; l < b.contrast.rows; l++) {
      double d = 0;
      for (std::size_t j = 0; j < b.parts; j++) d += b.contrast(k, j) * b.contrast(l, j);
      o = std::max(o, std::fabs(d - (k == l ? 1.0 : 0.0)));
    }
  }
  return {o, s};
}

double euclid(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0;
  for (std::size_t i = 0; i < a.size(); i++) s += (a[i] - b[i]) * (a[i] - b[i]);
  return std::sqrt(s);
}

std::vector<double> parse_list(const std::string& s, char sep = ',') {
  std::vector<double> v;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, sep)) v.push_back(std::stod(item));
  return v;
}

}  // namespace

int main() {
  calx::verification::Suite s;
  calx::random::Stream rng(20261113, 0x636f6d70, 1, 0);

  // ---- closure and its invariance ----
  {
    const std::vector<double> day{480, 420, 300, 120, 90, 30};
    const auto c1 = closure(day), c100 = closure(day, 100), c1440 = closure(day, 1440);
    s.check("closure to 1, 100 and 1440: the sums", std::fabs(std::accumulate(c1.begin(), c1.end(), 0.0) - 1) + std::fabs(std::accumulate(c100.begin(), c100.end(), 0.0) - 100) / 100 + std::fabs(std::accumulate(c1440.begin(), c1440.end(), 0.0) - 1440) / 1440, 4e-16);
    s.check("closure keeps the ratios: C₁₄₄₀ of minutes that sum to 1440 is the minutes", gap(c1440, day), 1e-12);
    double worst = 0;
    for (int t = 0; t < 50; t++) {
      const auto x = random_composition(rng, 5), y = random_composition(rng, 5);
      const double lx = std::exp(3 * rng.normal()), ly = std::exp(3 * rng.normal());
      std::vector<double> X(x), Y(y);
      for (double& v : X) v *= lx;
      for (double& v : Y) v *= ly;
      worst = std::max({worst, gap(perturb(x, y), perturb(X, Y)), gap(power(x, 0.37), power(X, 0.37)), gap(clr(x), clr(X)),
                        std::fabs(inner(x, y) - inner(X, Y)), std::fabs(distance(x, y) - distance(X, Y)), gap(ilr(x), ilr(X)),
                        gap(difference(x, y), difference(X, Y))});
    }
    s.check("perturbation, powering, difference, clr, ilr, inner product and distance invariant to the scale of their arguments (50 pairs)", worst, 1e-12);
    const auto x = random_composition(rng, 4), y = random_composition(rng, 4);
    const auto p1 = perturb(x, y, 1), p1440 = perturb(x, y, 1440);
    std::vector<double> p1s(p1);
    for (double& v : p1s) v *= 1440;
    s.check("an operation closed to 1440 is the one closed to 1, times 1440", gap(p1s, p1440) / 1440, 1e-15);
  }

  // ---- clr ----
  {
    double sum = 0, inv = 0;
    for (int t = 0; t < 50; t++) {
      const auto x = random_composition(rng, 2 + rng.below(9));
      const auto z = clr(x);
      double zs = 0;
      for (double v : z) zs += v;
      sum = std::max(sum, std::fabs(zs));
      inv = std::max(inv, gap(clr_inverse(z), closure(x)));
    }
    s.check("clr components sum to zero (50 compositions of 2 to 10 parts)", sum, 1e-14);
    s.check("clr⁻¹(clr x) = C(x)", inv, 1e-15);
    std::vector<double> z{0.3, -1.2, 5.0, 2.0};
    auto w = z;
    for (double& v : w) v += 17.5;
    s.check("clr⁻¹ ignores a constant added to its argument", gap(clr_inverse(z), clr_inverse(w)), 1e-15);
    const std::vector<double> big{800, 0, -800};
    const auto e = clr_inverse(big);
    s.require("clr⁻¹ neither overflows nor underflows at ±800", std::isfinite(e[0]) && e[0] == 1.0 && e[2] == 0.0 && std::isfinite(e[1]));
  }

  // ---- perturbation is a group, powering distributes ----
  {
    double neutral_gap = 0, inverse_gap = 0, assoc = 0, comm = 0, dist1 = 0, dist2 = 0, compo = 0, unit = 0, diff = 0;
    for (int t = 0; t < 40; t++) {
      const std::size_t D = 3 + rng.below(5);
      const auto x = random_composition(rng, D), y = random_composition(rng, D), z = random_composition(rng, D);
      const double a = rng.normal() * 2, b = rng.normal() * 2;
      neutral_gap = std::max(neutral_gap, gap(perturb(x, neutral(D)), closure(x)));
      inverse_gap = std::max(inverse_gap, gap(perturb(x, perturb_inverse(x)), neutral(D)));
      assoc = std::max(assoc, gap(perturb(perturb(x, y), z), perturb(x, perturb(y, z))));
      comm = std::max(comm, gap(perturb(x, y), perturb(y, x)));
      dist1 = std::max(dist1, gap(power(perturb(x, y), a), perturb(power(x, a), power(y, a))));
      dist2 = std::max(dist2, gap(power(x, a + b), perturb(power(x, a), power(x, b))));
      compo = std::max(compo, gap(power(x, a * b), power(power(x, b), a)));
      unit = std::max(unit, gap(power(x, 1), closure(x)));
      diff = std::max(diff, gap(difference(x, y), perturb(x, power(y, -1))));
    }
    s.check("perturbation: x ⊕ n = C(x)", neutral_gap, 1e-15);
    s.check("perturbation: x ⊕ (⊖x) = n", inverse_gap, 1e-15);
    s.check("perturbation: associative", assoc, 1e-15);
    s.check("perturbation: commutative", comm, 1e-15);
    s.check("powering distributes over perturbation: a ⊙ (x ⊕ y) = (a ⊙ x) ⊕ (a ⊙ y)", dist1, 1e-14);
    s.check("powering: (a + b) ⊙ x = (a ⊙ x) ⊕ (b ⊙ x)", dist2, 1e-14);
    s.check("powering: (ab) ⊙ x = a ⊙ (b ⊙ x)", compo, 1e-14);
    s.check("powering: 1 ⊙ x = C(x)", unit, 1e-15);
    s.check("difference: x ⊖ y = x ⊕ ((−1) ⊙ y)", diff, 1e-15);
  }

  // ---- inner product, norm, distance ----
  {
    double forms = 0, nrm = 0, dnorm = 0, inv = 0, scale = 0, linear = 0;
    for (int t = 0; t < 40; t++) {
      const std::size_t D = 2 + rng.below(7);
      const auto x = random_composition(rng, D), y = random_composition(rng, D), z = random_composition(rng, D);
      double dbl = 0;  // (1/(2D)) Σ_i Σ_j ln(x_i/x_j) ln(y_i/y_j)
      for (std::size_t i = 0; i < D; i++)
        for (std::size_t j = 0; j < D; j++) dbl += std::log(x[i] / x[j]) * std::log(y[i] / y[j]);
      dbl /= 2.0 * static_cast<double>(D);
      forms = std::max(forms, std::fabs(dbl - inner(x, y)) / (1 + std::fabs(dbl)));
      nrm = std::max(nrm, std::fabs(norm(x) * norm(x) - inner(x, x)) / (1 + inner(x, x)));
      dnorm = std::max(dnorm, std::fabs(distance(x, y) - norm(difference(x, y))));
      inv = std::max(inv, std::fabs(distance(perturb(x, z), perturb(y, z)) - distance(x, y)));
      const double a = 3 * rng.normal();
      scale = std::max(scale, std::fabs(distance(power(x, a), power(y, a)) - std::fabs(a) * distance(x, y)) / (1 + std::fabs(a)));
      linear = std::max(linear, std::fabs(inner(perturb(x, z), y) - inner(x, y) - inner(z, y)) + std::fabs(inner(power(x, a), y) - a * inner(x, y)));
    }
    s.check("inner product: (1/(2D)) Σ_ij ln(x_i/x_j) ln(y_i/y_j) = clr(x) · clr(y)", forms, 1e-13);
    s.check("‖x‖_a² = ⟨x, x⟩_a", nrm, 1e-14);
    s.check("d_a(x, y) = ‖x ⊖ y‖_a", dnorm, 1e-13);
    s.check("d_a invariant under perturbation: d(x ⊕ z, y ⊕ z) = d(x, y)", inv, 1e-13);
    s.check("d_a scales under powering: d(a ⊙ x, a ⊙ y) = |a| d(x, y)", scale, 1e-13);
    s.check("the inner product is bilinear in Aitchison's operations", linear, 1e-12);
    const std::vector<double> x{1, 2, 4};
    // clr(x) = ln 2 · (−1, 0, 1): ‖x‖² = 2 ln² 2
    s.check("‖(1, 2, 4)‖_a = √2 ln 2 by hand", norm(x) - std::sqrt(2.0) * std::log(2.0), 1e-15);
  }

  // ---- alr ----
  {
    const auto x = random_composition(rng, 5);
    s.check("alr⁻¹(alr x) = C(x), the last part as denominator", gap(alr_inverse(alr(x)), closure(x)), 1e-15);
    s.check("alr⁻¹(alr x) = C(x), the second part as denominator", gap(alr_inverse(alr(x, 1), 1), closure(x)), 1e-15);
    s.check("alr: ln(x₀/x₄) by hand", alr(x)[0] - std::log(x[0] / x[4]), 1e-15);
  }

  // ---- balances: orthonormal bases ----
  {
    double orth = 0, sums = 0;
    int bases = 0;
    for (std::size_t D = 2; D <= 8; D++)
      for (std::size_t l = 0; l < D; l++) {
        const auto [o, r] = orthonormality(pivot_basis(D, l));
        orth = std::max(orth, o), sums = std::max(sums, r), bases++;
      }
    s.check("pivot bases of 2 to 8 parts, every part first (35 bases): ΨΨᵀ = I", orth, 1e-15);
    s.check("pivot bases: Ψ1 = 0", sums, 1e-15);
    orth = sums = 0;
    for (int t = 0; t < 60; t++) {
      const std::size_t D = 2 + rng.below(11);
      const auto [o, r] = orthonormality(basis(random_partition(rng, D)));
      orth = std::max(orth, o), sums = std::max(sums, r);
    }
    s.check("random sequential binary partitions of 2 to 12 parts (60): ΨΨᵀ = I", orth, 1e-15);
    s.check("random partitions: Ψ1 = 0", sums, 1e-15);
    // A hand-written partition of six activities: (sleep, sedentary | light, moderate, vigorous, other) …
    const SignTable day{{1, 1, -1, -1, -1, -1}, {1, -1, 0, 0, 0, 0}, {0, 0, 1, -1, -1, -1}, {0, 0, 0, 1, -1, -1}, {0, 0, 0, 0, 1, -1}};
    const auto bd = basis(day);
    s.check("the first contrast of a 2-against-4 split: √(4/(2·6)) on the +1 parts, −√(2/(4·6)) on the −1 parts",
            std::fabs(bd.contrast(0, 0) - std::sqrt(4.0 / 12)) + std::fabs(bd.contrast(0, 5) + std::sqrt(2.0 / 24)), 1e-16);
  }

  // ---- ilr: isometry, linearity, inverse, balances ----
  {
    double iso = 0, dot = 0, inv = 0, lin = 0, bal = 0, piv = 0;
    for (int t = 0; t < 60; t++) {
      const std::size_t D = 2 + rng.below(9);
      const auto b = t % 2 ? basis(random_partition(rng, D)) : pivot_basis(D, rng.below(D));
      const auto x = random_composition(rng, D), y = random_composition(rng, D);
      const auto zx = ilr(x, b), zy = ilr(y, b);
      iso = std::max(iso, std::fabs(distance(x, y) - euclid(zx, zy)) / (1 + distance(x, y)));
      double d = 0;
      for (std::size_t k = 0; k < zx.size(); k++) d += zx[k] * zy[k];
      dot = std::max(dot, std::fabs(inner(x, y) - d) / (1 + std::fabs(d)));
      inv = std::max(inv, gap(ilr_inverse(zx, b), closure(x)));
      const double a = 2 * rng.normal();
      const auto zs = ilr(perturb(x, power(y, a)), b);
      std::vector<double> want(zx.size());
      for (std::size_t k = 0; k < zx.size(); k++) want[k] = zx[k] + a * zy[k];
      lin = std::max(lin, gap(zs, want));
      for (std::size_t k = 0; k + 1 < D; k++) {
        std::vector<std::size_t> plus, minus;
        for (std::size_t j = 0; j < D; j++) {
          if (b.partition[k][j] == 1) plus.push_back(j);
          if (b.partition[k][j] == -1) minus.push_back(j);
        }
        bal = std::max(bal, std::fabs(balance(x, plus, minus) - zx[k]));
      }
      // Hron et al.: z₁ = √((D − 1)/D) ln(x₁ / g(x₂, …, x_D)) for the parts in order.
      const auto zp = ilr(x);
      double lg = 0;
      for (std::size_t j = 1; j < D; j++) lg += std::log(x[j]);
      lg /= static_cast<double>(D - 1);
      piv = std::max(piv, std::fabs(zp[0] - std::sqrt((D - 1.0) / D) * (std::log(x[0]) - lg)));
    }
    s.check("ilr is an isometry: d_a(x, y) = ‖ilr x − ilr y‖ (60 pairs, pivot and random bases)", iso, 1e-13);
    s.check("ilr carries the inner product: ⟨x, y⟩_a = ilr x · ilr y", dot, 1e-13);
    s.check("ilr⁻¹(ilr x) = C(x)", inv, 1e-15);
    s.check("ilr is linear: ilr(x ⊕ (a ⊙ y)) = ilr x + a ilr y", lin, 1e-13);
    s.check("each balance √(rs/(r + s)) ln(g₊/g₋) equals its ilr coordinate", bal, 1e-13);
    s.check("the first pivot coordinate is √((D − 1)/D) ln(x₁/g(x₂, …, x_D)) (Hron, Filzmoser and Thompson 2012)", piv, 1e-13);
    // A change of basis is orthogonal and carries coordinates.
    const auto a = pivot_basis(6, 2), b = basis(random_partition(rng, 6));
    const auto R = change_of_basis(a, b);
    double o = 0;
    for (std::size_t i = 0; i < 5; i++)
      for (std::size_t j = 0; j < 5; j++) {
        double d = 0;
        for (std::size_t k = 0; k < 5; k++) d += R(i, k) * R(j, k);
        o = std::max(o, std::fabs(d - (i == j)));
      }
    s.check("a change of basis Ψ_b Ψ_aᵀ is orthogonal", o, 1e-15);
    const auto x = random_composition(rng, 6);
    s.check("and carries coordinates: ilr_b(x) = R ilr_a(x)", gap(calx::linalg::multiply(R, ilr(x, a)), ilr(x, b)), 1e-13);
    // Two parts: one coordinate, ln(x₁/x₂)/√2.
    const std::vector<double> two{3, 12};
    s.check("two parts: ilr = ln(x₁/x₂)/√2", ilr(two)[0] - std::log(0.25) / std::sqrt(2.0), 1e-15);
  }

  // ---- refusals ----
  {
    const std::vector<double> zero{0.5, 0.0, 0.5}, x{1, 2, 3};
    s.require("a zero part is refused by clr, with the reason", throws([&] { clr(zero); }));
    s.require("a zero part is refused by perturbation and powering", throws([&] { perturb(zero, x); }) && throws([&] { power(zero, 2); }));
    s.require("closure allows zeros but not a vanishing sum", !throws([&] { closure(zero); }) && throws([&] { closure(std::vector<double>{0, 0}); }));
    s.require("a partition with the wrong number of rows is refused", throws([&] { basis(SignTable{{1, -1, -1}}); }));
    s.require("a first row that leaves parts out is refused", throws([&] { basis(SignTable{{1, -1, 0}, {1, 1, -1}}); }));
    s.require("a row that splits no group made before it is refused", throws([&] { basis(SignTable{{1, 1, -1, -1}, {1, 0, -1, 0}, {0, 1, 0, -1}}); }));
    s.require("a row without a −1 is refused", throws([&] { basis(SignTable{{1, 1, 1}, {1, -1, 0}}); }));
    s.require("a sign other than ±1, 0 is refused", throws([&] { basis(SignTable{{1, -2, -1}, {0, 1, -1}}); }));
    s.require("rows out of their order (a split before its group exists) are refused", throws([&] { basis(SignTable{{0, 1, -1}, {1, -1, -1}}); }));
    s.require("the balance of overlapping groups is refused", throws([&] { balance(x, std::vector<std::size_t>{0, 1}, std::vector<std::size_t>{1, 2}); }));
    s.require("a pivot order that is no permutation is refused", throws([&] { pivot_partition(3, std::vector<std::size_t>{0, 0, 1}); }));
  }

  // ---- cross-checks against scikit-bio ----
  {
    std::ifstream in(std::string(TYCHE_FIXTURES) + "/skbio_composition.txt");
    if (!in) {
      s.skip("scikit-bio composition fixture not found");
    } else {
      std::map<std::string, std::vector<double>> comps;
      std::map<std::string, SignTable> sbps;
      std::string line;
      double ops = 0, ilr_gap = 0;
      int rows = 0;
      while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string kind, name, list;
        ls >> kind >> name >> list;
        if (kind == "composition") {
          comps[name] = parse_list(list);
          continue;
        }
        if (kind == "sbp") {
          SignTable t;
          std::stringstream rs(list);
          std::string r;
          while (std::getline(rs, r, ';')) {
            std::vector<int> row;
            for (double v : parse_list(r)) row.push_back(static_cast<int>(v));
            t.push_back(row);
          }
          sbps[name] = t;
          continue;
        }
        if (!comps.count(name)) continue;  // the sample and zero rows belong to check_composition_sample
        const auto& x = comps.at(name);
        std::vector<double> rev(x.rbegin(), x.rend());
        const auto want = parse_list(list);
        std::vector<double> got;
        if (kind == "closure") got = closure(x);
        else if (kind == "clr") got = clr(x);
        else if (kind == "clr_inv") {
          auto z = clr(x);
          for (std::size_t i = 0; i < z.size(); i++) z[i] += 0.25 * static_cast<double>(i);
          got = clr_inverse(z);
        } else if (kind == "power") got = power(x, 0.7);
        else if (kind == "perturb") got = perturb(x, rev);
        else if (kind == "perturb_inv") got = difference(x, rev);
        else if (kind == "inner") got = {inner(x, rev)};
        else if (kind == "ilr") {
          ilr_gap = std::max(ilr_gap, gap(ilr(x, basis(sbps.at(name))), want));
          rows++;
          continue;
        } else continue;
        double rel = 0;
        for (std::size_t i = 0; i < want.size() && i < got.size(); i++) rel = std::max(rel, std::fabs(got[i] - want[i]) / (1 + std::fabs(want[i])));
        if (got.size() != want.size()) rel = INFINITY;
        ops = std::max(ops, rel);
        rows++;
      }
      s.check("cross-check (scikit-bio 0.7.4): closure, clr, clr⁻¹, powering, perturbation, difference and the inner product of three compositions", ops, 1e-14);
      s.check("cross-check (scikit-bio 0.7.4): ilr in the bases of three sequential binary partitions (sbp_basis)", ilr_gap, 1e-14);
      s.require("cross-check: every fixture row read", rows == 24, std::to_string(rows) + " rows");
    }
  }

  return s.done();
}
